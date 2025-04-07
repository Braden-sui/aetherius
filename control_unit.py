import time
import json
import logging
import uuid
import traceback
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

# Google Cloud imports
from google.cloud import storage, firestore
import google.cloud.firestore
from google.cloud import aiplatform
from google.generative_ai.types import GenerationConfig
from google.cloud.aiplatform_v1beta1.types import CachedContent

# Import actual component implementations
from long_term_memory import LongTermMemory, NodeProperties, ProcessedData, UpdatePayload
from information_synthesizer import InformationSynthesizer, ContentType, EmotionClassifier
from tool_user import ToolUser, ToolType
from working_memory import WorkingMemory, MemorySection
from utils.logging_utils import setup_structured_logging

# Initialize structured logging
logger = setup_structured_logging("control_unit")

# --- Enums and Data Classes ---
class ActionType(Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    GOAL_MANAGEMENT = "goal_management"
    MEMORY_QUERY = "memory_query" # Changed from UPDATE to QUERY for retrieval focus
    MEMORY_SYNTHESIS = "memory_synthesis" # Added for triggering IS processing
    RESOURCE_CHECK = "resource_check" # Likely less frequent as an action, more internal check
    ERROR_HANDLING = "error_handling"
    AWAIT_OVERSIGHT = "await_oversight" # Explicit state/action
    STATE_CHECKPOINT = "state_checkpoint"
    HUMAN_INTERACTION = "human_interaction" # Specific action for HITL processing

class MemorySection(Enum):
    """Sections of working memory that can be formatted for prompts."""
    ACTIVE_GOAL = "active_goal"
    RECENT_ACTIONS = "recent_actions"
    CONTEXT_ITEMS = "context_items"
    RESOURCE_STATUS = "resource_status"
    LTM_RETRIEVALS = "ltm_retrievals"
    REASONING = "reasoning"
    TOOL_OUTPUTS = "tool_outputs"
    STATE_FLAGS = "state_flags"

class Goal:
    """Represents a goal for the agent."""
    def __init__(self, description: str, priority: float = 0.5, id: Optional[str] = None, status: str = "active", sub_goals: Optional[List['Goal']] = None, context_pointers: Optional[List[str]] = None):
        self.id = id or str(uuid.uuid4())
        self.description = description
        self.priority = float(priority)
        self.status = status # e.g., active, completed, failed, paused
        self.sub_goals = sub_goals or []
        self.context_pointers = context_pointers or [] # Links to relevant LTM nodes/data
        self.created_at = datetime.now().isoformat()

    def dict(self):
        return {
            "id": self.id, "description": self.description, "priority": self.priority,
            "status": self.status, "created_at": self.created_at,
            "sub_goals": [sg.dict() for sg in self.sub_goals], # Recursive dict conversion
            "context_pointers": self.context_pointers,
        }

class Action:
    """Represents an action to be taken by the agent."""
    def __init__(self, type: ActionType, description: str = "", params: Optional[Dict[str, Any]] = None, id: Optional[str] = None):
        self.id = id or str(uuid.uuid4()) # Unique ID for tracking, esp. for oversight
        self.type = type
        self.description = description
        self.params = params or {}
        self.timestamp = datetime.now().isoformat()
        self.status = "pending" # e.g., pending, executing, completed, failed, awaiting_oversight
        self.result: Optional[Dict[str, Any]] = None

    def dict(self):
        return {
            "id": self.id, "type": self.type.value, "description": self.description,
            "params": self.params, "timestamp": self.timestamp, "status": self.status,
            "result": self.result # Include result if available
        }

class ResourceUsage:
    """Tracks resource consumption."""
    def __init__(self, start_time: Optional[float] = None):
        self.total_cost_usd: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.cycle_count: int = 0
        self.start_time: float = start_time or time.time()

    def dict(self):
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cycle_count": self.cycle_count, "start_time": self.start_time,
            "elapsed_time": time.time() - self.start_time,
        }

class SystemState:
    """Encapsulates the entire state of the agent."""
    def __init__(self):
        self.working_memory = WorkingMemory()  # Use actual WorkingMemory implementation
        self.goals: List[Goal] = []
        self.resources: ResourceUsage = ResourceUsage()
        self.current_action: Optional[Action] = None # The action selected for the current cycle
        self.status: str = "initializing" # e.g., initializing, running, paused, awaiting_oversight, terminated

    def get_active_goals(self) -> List[Goal]:
        return sorted(
            [g for g in self.goals if g.status == "active"],
            key=lambda g: g.priority, reverse=True
        )

    def dict(self):
        return {
            "working_memory": self.working_memory.dict(),
            "goals": [goal.dict() for goal in self.goals],
            "resources": self.resources.dict(),
            "current_action": self.current_action.dict() if self.current_action else None,
            "status": self.status,
        }

# --- Error Classes ---
class ControlUnitError(Exception): pass
class CycleError(ControlUnitError): pass
class ResourceExceededError(ControlUnitError): pass
class LLMError(ControlUnitError): pass
class ToolError(ControlUnitError): pass
class MemoryError(ControlUnitError): pass
class GoalError(ControlUnitError): pass
class OversightError(ControlUnitError): pass

# --- Main Control Unit Implementation ---
class ControlUnit:
    """
    Orchestrates the Aetherius agent's cognitive loop, managing state,
    goals, actions, resources, and interactions between modules,
    adhering to the white paper v1.2 design.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        initial_goal_description: str,
        initial_context: str = "",
        session_id: Optional[str] = None,
        checkpoint_frequency: int = 10, # Cycles between checkpoints
        resource_limits: Optional[Dict[str, float]] = None,
        bootstrap_ltm: bool = False, # Flag to indicate if LTM needs bootstrapping
    ):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.cycle_count = 0 # Use state.resources.cycle_count primarily
        self.checkpoint_frequency = checkpoint_frequency

        logger.info(f"Initializing Control Unit for session: {self.session_id}", extra={"session_id": self.session_id})

        # Default resource limits (aligned with white paper examples)
        self.resource_limits = {
            "total_budget_usd": 10.0,
            "max_total_tokens": 1_000_000,
            "max_cycles": 500, # Increased default max cycles
            "max_cycle_time_sec": 60, # Max time allowed per cycle
            "target_cycle_time_sec": 20, # Target average cycle time (15-25s range)
            **(resource_limits or {}) # Allow overrides from config
        }
        logger.info(f"Resource Limits: {self.resource_limits}", extra={"session_id": self.session_id})

        # Initialize system state
        self.state = SystemState()
        self.state.status = "initializing"
        if initial_context:
            self.state.working_memory.add_context(f"Initial Context: {initial_context}")
        self.state.goals.append(Goal(description=initial_goal_description, priority=1.0))

        # Initialize components
        self._init_gcp_clients()
        self._init_ai_models()
        self._init_memory_systems(bootstrap_ltm)
        self._init_interaction_modules()
        self._create_or_refresh_context_cache() # Create initial cache

        # Initialize state persistence last, after components are ready
        self._init_state_persistence()

        self.state.status = "ready"
        logger.info("Control Unit initialized successfully.", extra={"session_id": self.session_id})
        self._checkpoint_state("initialization") # Initial checkpoint

    def _init_gcp_clients(self):
        """Initialize core GCP service clients."""
        logger.debug("Initializing GCP clients...")
        try:
            self.project_id = self.config["gcp"]["project_id"]
            self.location = self.config["gcp"]["location"]
            aiplatform.init(project=self.project_id, location=self.location)
            self.storage_client = storage.Client(project=self.project_id)
            self.firestore_client = firestore.Client(project=self.project_id)
            # Using v1beta1 client for CachedContent features
            self.cached_content_client = aiplatform.gapic.CachedContentServiceClient(
                 client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            )
            logger.debug("GCP clients initialized.")
        except Exception as e:
            logger.exception("Failed to initialize GCP clients.", exc_info=True)
            raise ControlUnitError("GCP Client Initialization Failed") from e

    def _init_ai_models(self):
        """Initialize generative models."""
        logger.debug("Initializing AI models...")
        try:
            # Default: Flash for CU reasoning (evaluation, selection)
            self.cu_llm_model_name = self.config.get("cu_llm_model", "gemini-1.5-flash-001")
            self.cu_llm = aiplatform.GenerativeModel(
                self.cu_llm_model_name,
                 generation_config=GenerationConfig(**self.config.get("cu_llm_config", {"temperature": 0.2, "max_output_tokens": 2048}))
            )

            # Core Reasoning: Pro for complex tasks, IE
            self.core_llm_model_name = self.config.get("core_llm_model", "gemini-1.5-pro-001")
            self.core_llm = aiplatform.GenerativeModel(
                self.core_llm_model_name,
                 generation_config=GenerationConfig(**self.config.get("core_llm_config", {"temperature": 0.5, "max_output_tokens": 8192}))
            )
            logger.debug(f"AI models initialized: CU ({self.cu_llm_model_name}), Core ({self.core_llm_model_name})")
        except Exception as e:
            logger.exception("Failed to initialize AI models.", exc_info=True)
            raise ControlUnitError("AI Model Initialization Failed") from e

    def _init_memory_systems(self, bootstrap_ltm: bool):
        """Initialize LTM and Information Synthesizer."""
        logger.debug("Initializing memory systems...")
        try:
            # LTM (using actual implementation)
            ltm_config = self.config.get('ltm', {})
            ltm_config["project_id"] = self.project_id  # Ensure project_id is passed
            ltm_config["location"] = self.location  # Ensure location is passed
            self.ltm = LongTermMemory(config=ltm_config)
            
            if bootstrap_ltm:
                logger.info("Bootstrapping LTM...")
                # Bootstrap implementation would go here if needed
                # This could involve loading initial data into LTM

            # Information Synthesizer
            is_config = self.config.get('information_synthesizer', {})
            is_config["project_id"] = self.project_id
            is_config["location"] = self.location
            is_config["emotion_classifier"] = {
                "project_id": self.project_id,
                "location": self.location,
                "emotion_endpoint_name": self.config.get("emotion_endpoint_name", "goemotions-endpoint")
            }
            
            self.information_synthesizer = InformationSynthesizer(config=is_config)
            logger.debug("Memory systems initialized.")
        except Exception as e:
            logger.exception("Failed to initialize memory systems.", exc_info=True)
            raise ControlUnitError("Memory System Initialization Failed") from e

    def _init_interaction_modules(self):
        """Initialize Tool User and potentially others."""
        logger.debug("Initializing interaction modules...")
        try:
            # Tool User (using actual implementation)
            tu_config = self.config.get('tool_user', {})
            tu_config["project_id"] = self.project_id
            tu_config["location"] = self.location
            
            # Add specific tool configurations
            tu_config["search_config"] = self.config.get("search_config", {})
            tu_config["twitter_config"] = self.config.get("twitter_config", {})
            
            self.tool_user = ToolUser(config=tu_config)
            logger.debug("Interaction modules initialized.")
        except Exception as e:
            logger.exception("Failed to initialize interaction modules.", exc_info=True)
            raise ControlUnitError("Interaction Module Initialization Failed") from e

    def _init_state_persistence(self):
        """Initialize state persistence components (GCS, Firestore)."""
        logger.debug("Initializing state persistence...")
        try:
            # Checkpoints via GCS
            self.checkpoint_bucket_name = self.config["persistence"]["checkpoint_bucket"]
            self.checkpoint_bucket = self.storage_client.bucket(self.checkpoint_bucket_name)

            # Real-time state/comms via Firestore
            self.state_collection_name = self.config["persistence"]["firestore_state_collection"]
            self.state_collection = self.firestore_client.collection(self.state_collection_name)
            self.human_input_collection_name = self.config["persistence"]["firestore_human_input_collection"]
            self.agent_response_collection_name = self.config["persistence"]["firestore_agent_response_collection"]
            self.oversight_collection_name = self.config["persistence"]["firestore_oversight_collection"]
            logger.debug(f"State persistence initialized: Bucket ({self.checkpoint_bucket_name}), Firestore Root ({self.state_collection_name})")
        except Exception as e:
            logger.exception("Failed to initialize state persistence.", exc_info=True)
            raise ControlUnitError("State Persistence Initialization Failed") from e

    # --- Context Cache Management ---
    def _create_or_refresh_context_cache(self) -> None:
        """Creates or refreshes the Vertex AI Context Cache for the static system prompt."""
        try:
            system_prompt = self.config.get("system_prompt", "")
            if not system_prompt:
                logger.warning("No system_prompt found in config, cannot create context cache.", extra={"session_id": self.session_id})
                self.context_cache_resource_name = None
                self.cache_expiry_time = None
                return

            # Use the Core LLM (Pro) model name for cache association
            model_uri = f"publishers/google/models/{self.core_llm_model_name}"
            # TTL (e.g., 1 hour)
            ttl_seconds = self.config.get("context_cache_ttl_seconds", 3600)

            parent = f"projects/{self.project_id}/locations/{self.location}"
            cache_id = f"aetherius-persona-{self.session_id}-{int(time.time())}" # Unique ID

            # Using v1beta1 CachedContent type
            cached_content = CachedContent(
                display_name=cache_id,
                model=model_uri, # Associate with the specific model version
                system_instruction = aiplatform.gapic.Content(parts=[aiplatform.gapic.Part(text=system_prompt)]),
                ttl=timedelta(seconds=ttl_seconds),
            )

            logger.info(f"Attempting to create context cache for model {model_uri}...", extra={"session_id": self.session_id})
            # Use the v1beta1 client method
            response = self.cached_content_client.create_cached_content(
                parent=parent,
                cached_content=cached_content,
            )

            new_cache_resource_name = response.name
            new_cache_expiry_time = time.time() + ttl_seconds

            logger.info(f"Successfully created/refreshed context cache: {new_cache_resource_name}", extra={
                "session_id": self.session_id, "cache_name": new_cache_resource_name, "expires_at": datetime.fromtimestamp(new_cache_expiry_time).isoformat()
            })

            # Clean up old cache if it exists and is different
            old_cache_name = getattr(self, 'context_cache_resource_name', None)
            if old_cache_name and old_cache_name != new_cache_resource_name:
                try:
                    logger.warning(f"Deleting previous context cache: {old_cache_name}", extra={"session_id": self.session_id})
                    self.cached_content_client.delete_cached_content(name=old_cache_name)
                except Exception as delete_err:
                    logger.warning(f"Failed to delete previous context cache {old_cache_name}: {delete_err}", extra={"session_id": self.session_id})

            self.context_cache_resource_name = new_cache_resource_name
            self.cache_expiry_time = new_cache_expiry_time

        except Exception as e:
            logger.error(f"Error creating/refreshing context cache: {e}", exc_info=True, extra={"session_id": self.session_id})
            self.context_cache_resource_name = None
            self.cache_expiry_time = None # Force non-cached calls if creation fails

    def _check_and_refresh_context_cache(self) -> None:
        """Checks cache expiry and refreshes if necessary."""
        refresh_buffer_seconds = 600 # Refresh if within 10 minutes of expiry
        if not self.cache_expiry_time or time.time() > (self.cache_expiry_time - refresh_buffer_seconds):
            logger.info("Context cache expired or nearing expiry. Refreshing...", extra={"session_id": self.session_id})
            self._create_or_refresh_context_cache()

    # --- Core Operational Loop ---
    def run_cycle(self) -> SystemState:
        """Executes one full cognitive cycle."""
        cycle_start_time = time.time()
        self.state.resources.cycle_count += 1
        self.cycle_count = self.state.resources.cycle_count # Keep internal counter sync'd
        logger.info(f"--- Cycle {self.cycle_count} Start ---", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        self.state.status = "running"

        try:
            # 0. Check for external signals (Human Input, Oversight Approval)
            self._check_external_signals() # Handles HITL and oversight resolution

            # If paused for oversight, skip rest of cycle until resolved
            if self.state.status == "awaiting_oversight":
                 logger.warning(f"Cycle {self.cycle_count}: Awaiting oversight approval for action {self.state.current_action.id}. Skipping normal execution.",
                               extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": self.state.current_action.id})
                 # Short sleep to avoid busy-waiting? Or rely on external trigger?
                 time.sleep(5) # Simple polling delay
                 return self.state # Return current state

            # 1. Resource Check (Abort if limits exceeded)
            self._check_resources()

            # 2. State Evaluation (What's the situation?)
            evaluation = self._evaluate_state()
            self.state.working_memory.add_context(f"State Evaluation (Cycle {self.cycle_count}): {json.dumps(evaluation)}")

            # 3. Action Selection (What should I do next?)
            action = self._select_next_action(evaluation)
            self.state.current_action = action # Store selected action in state
            self.state.working_memory.add_action(action.dict()) # Add to history using the dict method
            logger.info(f"Selected Action: {action.type.value} - {action.description}",
                       extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id, "action_type": action.type.value})

            # 4. Handle Oversight Request (Pause if needed)
            if self._requires_oversight(action):
                 self._request_oversight(action)
                 # State is now 'awaiting_oversight', next cycle will check for approval
                 logger.warning(f"Cycle {self.cycle_count}: Action {action.id} requires oversight. Pausing.",
                                extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
                 return self.state # End cycle here, wait for approval check

            # 5. Action Execution (Do it!)
            action.status = "executing"
            result = self._execute_action(action)
            action.result = result # Store result within the action object
            action.status = result.get("status", "unknown") # e.g., success, error

            # 6. State Update (What changed?)
            self._update_state(action, result)

            # 7. Data Synthesis (Process new info for LTM) - Can happen async?
            self._trigger_information_synthesis()

            # 8. Checkpoint State Periodically
            if self.cycle_count % self.checkpoint_frequency == 0:
                self._checkpoint_state(f"cycle_{self.cycle_count}")

            # Cycle time logging
            cycle_duration = time.time() - cycle_start_time
            logger.info(f"--- Cycle {self.cycle_count} End ({cycle_duration:.2f}s) ---",
                       extra={"session_id": self.session_id, "cycle": self.cycle_count, "duration": cycle_duration})

            # Check cycle time against limits
            if cycle_duration > self.resource_limits["max_cycle_time_sec"]:
                 logger.warning(f"Cycle {self.cycle_count} exceeded max time: {cycle_duration:.2f}s > {self.resource_limits['max_cycle_time_sec']}s",
                                extra={"session_id": self.session_id, "cycle": self.cycle_count, "duration": cycle_duration})
                 self._implement_adaptive_tempo("latency") # Trigger tempo adaptation

            return self.state

        except ResourceExceededError as r_err:
            logger.error(f"Resource limit exceeded in Cycle {self.cycle_count}: {r_err}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count})
            self.state.status = "terminated_resource_limit"
            self._handle_termination(r_err) # Specific handling for resource limits
            raise # Re-raise to stop execution if desired

        except Exception as e:
            logger.error(f"Unhandled error in Cycle {self.cycle_count}: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count})
            self.state.status = "error"
            # Attempt general error handling within the state
            try:
                self._handle_general_error(e)
            except Exception as handler_err:
                 logger.critical(f"Error occurred *during* error handling: {handler_err}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count})
                 self.state.status = "critical_error"
                 self._checkpoint_state("critical_error") # Attempt final checkpoint
                 raise ControlUnitError("Critical failure during error handling") from handler_err
            return self.state # Return state after attempting recovery

    # --- Sub-steps of the Cycle ---

    def _check_resources(self) -> None:
        """Checks resource usage against defined limits."""
        res = self.state.resources
        limits = self.resource_limits

        # Log current usage for monitoring
        logger.debug(f"Resource Check: Cost=${res.total_cost_usd:.4f}/{limits['total_budget_usd']:.2f}, Tokens={res.total_input_tokens+res.total_output_tokens}/{limits['max_total_tokens']}, Cycles={res.cycle_count}/{limits['max_cycles']}",
                    extra={"session_id": self.session_id, "cycle": self.cycle_count, "resources": res.dict()})

        if res.total_cost_usd >= limits["total_budget_usd"]:
            raise ResourceExceededError(f"Budget exceeded: ${res.total_cost_usd:.2f} >= ${limits['total_budget_usd']:.2f}")
        if (res.total_input_tokens + res.total_output_tokens) >= limits["max_total_tokens"]:
            raise ResourceExceededError(f"Total token limit exceeded: {res.total_input_tokens + res.total_output_tokens} >= {limits['max_total_tokens']}")
        if res.cycle_count >= limits["max_cycles"]:
            raise ResourceExceededError(f"Cycle limit exceeded: {res.cycle_count} >= {limits['max_cycles']}")

    def _implement_adaptive_tempo(self, reason: str = "latency") -> None:
        """Adjusts operational parameters to manage resources or latency."""
        logger.warning(f"Implementing adaptive tempo due to: {reason}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        # Example strategies (could be made more sophisticated):
        # 1. Reduce LLM usage complexity/frequency
        # 2. Increase checkpoint frequency? Or decrease?
        # 3. Defer optional tasks like deep LTM reflection
        # 4. Limit breadth of tool searches
        self.state.working_memory.add_context(f"ADAPTIVE_TEMPO: Triggered due to {reason}. Adjusting strategy.")
        # For now, just logs and adds context. Real implementation needs more logic.

    def _evaluate_state(self) -> Dict[str, Any]:
        """Uses CU LLM (Flash) to analyze the current situation."""
        logger.debug("Evaluating state...", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        context = self._prepare_context_for_llm(purpose="state_evaluation")
        prompt = self._create_state_evaluation_prompt(context)

        try:
            # Use the CU's designated LLM (likely Flash)
            response = self.cu_llm.generate_content(prompt) # No context cache for evaluation/selection prompts - they are dynamic
            self._update_resource_usage_from_response("cu_llm", response) # Track usage

            if not response.candidates: raise LLMError("Empty response from LLM during state evaluation")

            evaluation_text = response.candidates[0].text
            logger.debug(f"Raw state evaluation response: {evaluation_text}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
            evaluation = self._parse_llm_json_response(evaluation_text, default_if_error={})
            return evaluation

        except Exception as e:
            logger.error(f"Error during state evaluation LLM call: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count})
            # Fallback: return basic info
            active_goals = self.state.get_active_goals()
            return {
                "error": f"Failed to evaluate state: {e}",
                "active_goal_id": active_goals[0].id if active_goals else None,
                "next_action_suggestion": ActionType.ERROR_HANDLING.value, # Suggest error handling
            }

    def _select_next_action(self, evaluation: Dict[str, Any]) -> Action:
        """Uses CU LLM (Flash) to determine the best action based on evaluation."""
        logger.debug("Selecting next action...", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        context = self._prepare_context_for_llm(purpose="action_selection")
        prompt = self._create_action_selection_prompt(context, evaluation)

        try:
            response = self.cu_llm.generate_content(prompt) # No context cache here either
            self._update_resource_usage_from_response("cu_llm", response)

            if not response.candidates: raise LLMError("Empty response from LLM during action selection")

            action_text = response.candidates[0].text
            logger.debug(f"Raw action selection response: {action_text}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
            action_data = self._parse_llm_json_response(action_text, default_if_error={})

            # Validate and create Action object
            action_type_str = action_data.get("action_type")
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                 logger.warning(f"LLM selected invalid action type: {action_type_str}. Defaulting to ERROR_HANDLING.",
                                extra={"session_id": self.session_id, "cycle": self.cycle_count})
                 action_type = ActionType.ERROR_HANDLING
                 action_data["description"] = f"Invalid action type '{action_type_str}' selected by LLM."
                 action_data["params"] = {"original_selection": action_data}


            action = Action(
                type=action_type,
                description=action_data.get("description", "No description provided."),
                params=action_data.get("params", {})
            )
            return action

        except Exception as e:
            logger.error(f"Error during action selection LLM call: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count})
            # Fallback to error handling action
            return Action(
                type=ActionType.ERROR_HANDLING,
                description=f"Failed to select action due to error: {e}",
                params={"error": str(e)}
            )

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Dispatcher for executing the selected action."""
        logger.debug(f"Executing action: {action.type.value}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
        start_time = time.time()
        result = {"status": "error", "message": "Action type not implemented"} # Default result

        try:
            if action.type == ActionType.LLM_CALL:
                result = self._execute_llm_call(action)
            elif action.type == ActionType.TOOL_CALL:
                result = self._execute_tool_call(action)
            elif action.type == ActionType.GOAL_MANAGEMENT:
                result = self._execute_goal_management(action)
            elif action.type == ActionType.MEMORY_QUERY:
                result = self._execute_memory_query(action)
            elif action.type == ActionType.MEMORY_SYNTHESIS:
                result = self._execute_memory_synthesis(action)
            elif action.type == ActionType.ERROR_HANDLING:
                result = self._execute_error_handling(action)
            elif action.type == ActionType.STATE_CHECKPOINT:
                result = self._execute_state_checkpoint(action)
            elif action.type == ActionType.HUMAN_INTERACTION:
                 result = self._execute_human_interaction_response(action) # Specific handler for responding to human
            # AWAIT_OVERSIGHT is a state, not an executable action here
            # RESOURCE_CHECK is done internally, not usually an explicit action

            else:
                 logger.warning(f"Attempted to execute unknown or non-executable action type: {action.type.value}",
                                extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
                 result = {"status": "error", "message": f"Unknown or non-executable action type: {action.type.value}"}

            duration = time.time() - start_time
            logger.debug(f"Action {action.type.value} execution finished in {duration:.2f}s", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id, "duration": duration})

            # Merge potential resource usage from result into main tracking
            if "resource_usage" in result:
                 self._update_resource_usage(
                     cost_usd=result["resource_usage"].get("cost_usd", 0.0),
                     input_tokens=result["resource_usage"].get("input_tokens", 0),
                     output_tokens=result["resource_usage"].get("output_tokens", 0),
                 )

            return result

        except Exception as e:
            logger.error(f"Exception during execution of action {action.type.value} ({action.id}): {e}", exc_info=True,
                        extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            return {"status": "error", "message": f"Execution failed: {e}", "error_type": type(e).__name__}

    def _update_state(self, action: Action, result: Dict[str, Any]):
        """Updates WM, goals, etc., based on the action's result."""
        logger.debug(f"Updating state after action {action.type.value}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})

        # Always add action result (even errors) to WM context for traceability
        self.state.working_memory.add_context(f"Action Result ({action.type.value} - {action.status}): {json.dumps(result)}")

        if action.status == "success":
            # Add specific successful outputs to WM or mark for synthesis
            if action.type == ActionType.LLM_CALL and "response" in result:
                # Add to working memory
                self.state.working_memory.set_reasoning(result["response"])
                
                # Mark for synthesis
                self.state.working_memory.add_pending_synthesis(
                    (
                        {"text": result["response"], "thought_type": "reasoning"}, 
                        "internal_reasoning", 
                        {"action_id": action.id, "model_used": result.get("model_used")}
                    )
                )
                
            elif action.type == ActionType.TOOL_CALL and "result" in result:
                # Add tool result as observation
                tool_name = action.params.get("tool_name", "unknown_tool")
                self.state.working_memory.add_observation({
                    "summary": f"Tool {tool_name} result: {str(result['result'])[:100]}...",
                    "timestamp": datetime.now().isoformat(),
                    "tool_name": tool_name,
                    "result": result["result"]
                })
                 
                # Mark for synthesis
                self.state.working_memory.add_pending_synthesis(
                    (
                        result["result"], 
                        f"tool_{tool_name}", 
                        {"action_id": action.id}
                    )
                )
                 
            elif action.type == ActionType.MEMORY_QUERY and "retrieved_data" in result:
                 # Add retrieved LTM snippets to LTM retrievals section
                 self.state.working_memory.add_ltm_retrieval({
                     "query": action.params.get("query", "Unknown query"),
                     "query_type": action.params.get("query_type", "unknown"),
                     "timestamp": datetime.now().isoformat(),
                     "results": result["retrieved_data"]
                 })
                 
            elif action.type == ActionType.HUMAN_INTERACTION and "response_text" in result:
                 # Add agent response as observation
                 self.state.working_memory.add_observation({
                     "summary": f"Agent response to human: {result['response_text'][:100]}...",
                     "timestamp": datetime.now().isoformat(),
                     "type": "agent_response",
                     "human_input_id": action.params.get("human_input_id")
                 })
                 
                 # Mark interaction for synthesis
                 self.state.working_memory.add_pending_synthesis(
                     (
                         {"human_input": action.params.get("human_input_text"), "agent_response": result["response_text"]},
                         "human_interaction",
                         {"action_id": action.id, "human_input_id": action.params.get("human_input_id")}
                     )
                 )

        # Goal status updates might happen within _execute_goal_management
        # Resource usage updates handled by _execute_action wrapper

    def _trigger_information_synthesis(self):
        """Sends pending data from WM to the Information Synthesizer."""
        pending = self.state.working_memory.pending_data_for_synthesis
        if not pending:
            return

        logger.info(f"Triggering information synthesis for {len(pending)} item(s)...", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        for data, source, metadata in pending:
            try:
                # Add cycle info to metadata
                metadata["cycle"] = self.cycle_count
                metadata["session_id"] = self.session_id
                
                # Determine content type based on source
                content_type = ContentType.LLM_THOUGHT  # Default content type
                if source.startswith("tool_google_search"):
                    content_type = ContentType.SEARCH_RESULT
                elif source.startswith("tool_twitter"):
                    content_type = ContentType.TWEET
                elif source == "internal_reasoning":
                    content_type = ContentType.LLM_THOUGHT
                elif source == "imagination":
                    content_type = ContentType.IMAGINATION
                elif source == "human_interaction":
                    content_type = ContentType.INTERACTION
                elif source == "decision":
                    content_type = ContentType.DECISION
                    
                # Process the content through the Information Synthesizer
                processed_data = self.information_synthesizer.process_content(
                    content=data,
                    content_type=content_type
                )
                
                # Store the processed data in Long-Term Memory
                # Convert processed_data to format expected by LTM
                graph_data = {
                    "nodes": processed_data.get("graph_data", {}).get("nodes", []),
                    "relationships": processed_data.get("graph_data", {}).get("relationships", [])
                }
                
                vector_data = None
                if processed_data.get("vector_data"):
                    vector_data = {
                        "id": processed_data["vector_data"].get("id", ""),
                        "embedding": processed_data["vector_data"].get("embedding", [])
                    }
                
                ltm_data = ProcessedData(
                    graph_data=graph_data,
                    vector_data=vector_data
                )
                
                store_success = self.ltm.store(ltm_data)
                
                if not store_success:
                    logger.warning(f"Failed to store processed data in LTM for source {source}", 
                                  extra={"session_id": self.session_id, "cycle": self.cycle_count})
                    
            except Exception as e:
                logger.error(f"Error during information synthesis for source {source}: {e}", exc_info=True,
                            extra={"session_id": self.session_id, "cycle": self.cycle_count, "source": source})

        # Clear the pending list after attempting synthesis
        self.state.working_memory.pending_data_for_synthesis = []

    # --- Action Execution Implementations ---

    def _execute_llm_call(self, action: Action) -> Dict[str, Any]:
        """Executes a call to one of the generative models."""
        prompt = action.params.get("prompt", "")
        model_choice = action.params.get("model", "core") # 'core' (Pro) or 'cu' (Flash)
        use_cache = action.params.get("use_cache", True) # Default to using cache for Core LLM

        # Select model instance
        if model_choice == "core":
            model = self.core_llm
            model_name = self.core_llm_model_name
            # Only Core LLM uses the static context cache
            cache_name_to_use = self.context_cache_resource_name if use_cache else None
            if use_cache: self._check_and_refresh_context_cache() # Ensure cache is fresh if requested
        else: # Default to CU LLM (Flash)
            model = self.cu_llm
            model_name = self.cu_llm_model_name
            cache_name_to_use = None # CU LLM calls (eval, select) don't use static cache

        logger.debug(f"Executing LLM call using {model_name} (Cache: {cache_name_to_use is not None})",
                    extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})

        try:
            # Construct generation config from action params or defaults
            gen_config_params = action.params.get("generation_config", {})
            # Get defaults from the initialized model's config if not overridden
            final_gen_config = GenerationConfig(**{**model.generation_config._asdict(), **gen_config_params})

            response = model.generate_content(
                prompt,
                generation_config=final_gen_config,
                cached_content = cache_name_to_use # Pass cache name if applicable
            )

            # Track resource usage (will be merged by caller)
            resource_usage = self._calculate_resource_usage("llm", response)

            if not response.candidates: raise LLMError(f"LLM call to {model_name} returned no candidates.")

            response_text = response.text # Use .text for simple aggregation

            return {
                "status": "success",
                "response": response_text,
                "model_used": model_name,
                "resource_usage": resource_usage
            }

        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            raise LLMError(f"LLM call failed: {e}") from e # Re-raise as specific error type

    def _execute_tool_call(self, action: Action) -> Dict[str, Any]:
        """Executes a tool using the ToolUser module."""
        tool_name = action.params.get("tool_name")
        parameters = action.params.get("params", {})
        if not tool_name: raise ToolError("Tool name not specified in action parameters.")

        logger.debug(f"Executing tool: {tool_name}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})

        try:
            # Convert tool_name string to ToolType enum
            try:
                tool_type = ToolType[tool_name.upper()]
            except (KeyError, AttributeError):
                raise ToolError(f"Unknown tool type: {tool_name}")
                
            # Execute the tool with the proper ToolType enum
            tool_response = self.tool_user.execute_tool(
                tool_type=tool_type,
                params=parameters
            )

            # Check for tool execution errors
            if tool_response.get("status") != "success":
                raise ToolError(f"Tool '{tool_name}' execution failed: {tool_response.get('error', 'Unknown tool error')}")

            # Calculate resource usage (cost might be in response)
            resource_usage = self._calculate_resource_usage(f"tool_{tool_name}", tool_response)

            # Add tool output to working memory
            self.state.working_memory.add_tool_output({
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
                "summary": f"Tool {tool_name} output: {str(tool_response)[:100]}...",
                "result": tool_response
            })

            return {
                "status": "success",
                "tool_name": tool_name,
                "result": tool_response,
                "resource_usage": resource_usage
            }

        except Exception as e:
            logger.error(f"Tool call to '{tool_name}' failed: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            # Don't re-raise immediately, let the main loop handle it via status
            return {"status": "error", "message": f"Tool '{tool_name}' failed: {e}", "error_type": type(e).__name__}

    def _execute_goal_management(self, action: Action) -> Dict[str, Any]:
        """Adds, updates, or removes goals."""
        operation = action.params.get("operation") # e.g., add, complete, fail, update_priority, decompose
        goal_data = action.params.get("goal_data", {})

        logger.debug(f"Executing goal management: {operation}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})

        try:
            if operation == "add":
                if "description" not in goal_data: raise GoalError("Goal description missing for 'add' operation.")
                new_goal = Goal(**goal_data)
                self.state.goals.append(new_goal)
                logger.info(f"Added new goal: {new_goal.id} - {new_goal.description}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                
                # Update working memory with active goal if this is now the highest priority
                active_goals = self.state.get_active_goals()
                if active_goals and active_goals[0].id == new_goal.id:
                    self.state.working_memory.set_active_goal(new_goal.dict())
                
            elif operation in ["complete", "fail", "pause"]:
                goal_id = goal_data.get("id")
                if not goal_id: raise GoalError(f"Goal ID missing for '{operation}' operation.")
                goal_updated = False
                for goal in self.state.goals:
                    if goal.id == goal_id:
                        goal.status = operation
                        logger.info(f"Updated goal {goal_id} status to {operation}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                        goal_updated = True
                        break
                if not goal_updated: raise GoalError(f"Goal ID {goal_id} not found for '{operation}' operation.")
                
                # Update active goal in working memory if needed
                active_goals = self.state.get_active_goals()
                if active_goals:
                    self.state.working_memory.set_active_goal(active_goals[0].dict())
                else:
                    # No active goals, clear the active goal
                    self.state.working_memory.set_active_goal({})
                
            elif operation == "update_priority":
                goal_id = goal_data.get("id")
                new_priority = goal_data.get("priority")
                if not goal_id or new_priority is None: raise GoalError("Goal ID or priority missing for 'update_priority'.")
                goal_updated = False
                for goal in self.state.goals:
                     if goal.id == goal_id:
                         goal.priority = float(new_priority)
                         logger.info(f"Updated goal {goal_id} priority to {new_priority}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                         goal_updated = True
                         break
                if not goal_updated: raise GoalError(f"Goal ID {goal_id} not found for 'update_priority'.")
                
                # Re-sort and update active goal in working memory if needed
                active_goals = self.state.get_active_goals()
                if active_goals:
                    self.state.working_memory.set_active_goal(active_goals[0].dict())
                
            elif operation == "decompose":
                # Implementation for goal decomposition - requires LLM call to break down a goal
                parent_goal_id = goal_data.get("id")
                if not parent_goal_id: raise GoalError("Parent goal ID missing for 'decompose' operation.")
                
                # Find the parent goal
                parent_goal = None
                for goal in self.state.goals:
                    if goal.id == parent_goal_id:
                        parent_goal = goal
                        break
                
                if not parent_goal: raise GoalError(f"Parent goal ID {parent_goal_id} not found for 'decompose'.")
                
                # Use LLM to generate sub-goals
                sub_goals_description = goal_data.get("sub_goals_description", [])
                new_sub_goals = []
                
                if not sub_goals_description:
                    # If no sub_goals_description provided, we'd need to call the LLM to generate them
                    # This would involve preparing a prompt, calling the LLM, parsing the response, etc.
                    # For simplicity, we'll just raise an error here
                    raise GoalError("Sub-goals description missing for 'decompose' operation.")
                
                # Create sub-goals from the descriptions
                for description in sub_goals_description:
                    sub_goal = Goal(
                        description=description,
                        priority=parent_goal.priority * 0.9,  # Slightly lower priority than parent
                        status="active"
                    )
                    new_sub_goals.append(sub_goal)
                    self.state.goals.append(sub_goal)
                
                # Add sub-goals to parent goal
                parent_goal.sub_goals.extend(new_sub_goals)
                
                logger.info(f"Decomposed goal {parent_goal_id} into {len(new_sub_goals)} sub-goals", 
                           extra={"session_id": self.session_id, "cycle": self.cycle_count})
                
            else:
                raise GoalError(f"Unsupported goal management operation: {operation}")

            return {
                "status": "success", 
                "operation": operation,
                "goal_id": goal_data.get("id")
            }

        except Exception as e:
            logger.error(f"Goal management failed: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            return {"status": "error", "message": f"Goal management failed: {e}", "error_type": type(e).__name__}

    def _execute_memory_query(self, action: Action) -> Dict[str, Any]:
        """Queries the Long-Term Memory."""
        query_type_str = action.params.get("query_type", "vector") # 'vector', 'graph', 'hybrid', 'metadata', 'emotional'
        query = action.params.get("query")
        params = action.params.get("params", {}) # e.g., top_k for vector, graph params

        if not query: raise MemoryError("Query missing for memory_query action.")

        logger.debug(f"Executing memory query ({query_type_str})", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})

        try:
            # Map query_type_str to LongTermMemory.QueryType enum
            query_type_map = {
                "vector": LongTermMemory.QueryType.SEMANTIC,
                "graph": LongTermMemory.QueryType.RELATIONAL,
                "hybrid": LongTermMemory.QueryType.HYBRID_SEMANTIC_FIRST,
                "metadata": LongTermMemory.QueryType.METADATA,
                "emotional": LongTermMemory.QueryType.EMOTIONAL
            }
            
            query_type = query_type_map.get(query_type_str)
            if not query_type:
                raise MemoryError(f"Unsupported memory query type: {query_type_str}")
                
            # Prepare query params
            query_params = {
                "query": query,
                **params  # Include any additional params passed in the action
            }
            
            # Execute query
            results = self.ltm.query(query_params, query_type)
            
            # Add results to working memory
            self.state.working_memory.add_ltm_retrieval({
                "query": query,
                "query_type": query_type_str,
                "timestamp": datetime.now().isoformat(),
                "summary": f"Retrieved {len(results.get('results', []))} items from LTM",
                "results": results
            })

            return {"status": "success", "retrieved_data": results}

        except Exception as e:
            logger.error(f"Memory query failed: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            return {"status": "error", "message": f"Memory query failed: {e}", "error_type": type(e).__name__}

    def _execute_memory_synthesis(self, action: Action) -> Dict[str, Any]:
        """Explicitly triggers the Information Synthesizer for pending data."""
        # This action might be selected if the agent decides it needs to consolidate
        # its recent experiences before proceeding.
        logger.debug("Executing explicit memory synthesis trigger", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
        try:
            self._trigger_information_synthesis()
            return {"status": "success", "message": "Information synthesis triggered."}
        except Exception as e:
            logger.error(f"Explicit memory synthesis trigger failed: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            return {"status": "error", "message": f"Memory synthesis trigger failed: {e}", "error_type": type(e).__name__}

    def _execute_error_handling(self, action: Action) -> Dict[str, Any]:
        """Executes specific error recovery logic."""
        # This is called if the LLM explicitly selects ERROR_HANDLING.
        # The actual error might have already been logged by the main loop catcher.
        error_details = action.params.get("error", "No specific error details provided.")
        logger.warning(f"Executing explicit error handling action: {error_details}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
        # Add recovery goal if not already present
        if not any("error recovery" in g.description.lower() for g in self.state.goals if g.status=='active'):
             recovery_goal = Goal(description=f"Recover from error: {error_details}", priority=1.0) # High priority
             self.state.goals.append(recovery_goal)
             # Update working memory with new active goal
             self.state.working_memory.set_active_goal(recovery_goal.dict())
             logger.info("Added error recovery goal.", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        return {"status": "success", "message": "Error handling initiated, recovery goal added."}

    def _execute_state_checkpoint(self, action: Action) -> Dict[str, Any]:
        """Explicitly triggers a state checkpoint."""
        logger.debug("Executing explicit state checkpoint action", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
        try:
            reason = action.params.get("reason", f"explicit_action_{action.id}")
            checkpoint_id = self._checkpoint_state(reason)
            return {"status": "success", "checkpoint_id": checkpoint_id}
        except Exception as e:
            logger.error(f"Explicit state checkpoint failed: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            return {"status": "error", "message": f"State checkpoint failed: {e}", "error_type": type(e).__name__}

    def _execute_human_interaction_response(self, action: Action) -> Dict[str, Any]:
        """Generates and sends a response to a human input."""
        human_input_id = action.params.get("human_input_id")
        human_input_text = action.params.get("human_input_text")
        if not human_input_id or not human_input_text:
            raise ValueError("Missing human input ID or text for response action.")

        logger.debug(f"Generating response for human input {human_input_id}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})

        try:
            # 1. Prepare context (including the human input)
            context = self._prepare_context_for_llm(purpose="human_interaction")
            prompt = f"""
            {context}

            ---
            Human Input (ID: {human_input_id}):
            {human_input_text}
            ---

            Generate a helpful and contextually relevant response to the human input above.
            """

            # 2. Call Core LLM (Pro) for a better quality response
            llm_action_params = {
                "prompt": prompt,
                "model": "core", # Use Gemini Pro
                "use_cache": True, # Use system prompt cache
                "generation_config": {"temperature": 0.7} # Allow more creativity
            }
            llm_response_action = Action(type=ActionType.LLM_CALL, params=llm_action_params)
            llm_result = self._execute_llm_call(llm_response_action) # Execute nested action

            if llm_result["status"] == "success":
                response_text = llm_result["response"]
                # 3. Send response back via Firestore
                self._send_response_to_human(human_input_id, response_text)
                # Result for the outer action
                return {"status": "success", "response_text": response_text, "human_input_id": human_input_id}
            else:
                 raise LLMError(f"Failed to generate human response: {llm_result.get('message')}")

        except Exception as e:
            logger.error(f"Failed to execute human interaction response: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id})
            # Attempt to send an error message back to the human
            try:
                self._send_response_to_human(human_input_id, f"Sorry, I encountered an error trying to respond: {type(e).__name__}")
            except Exception as send_err:
                 logger.error(f"Failed even to send error response to human {human_input_id}: {send_err}", exc_info=True, extra={"session_id": self.session_id})
            return {"status": "error", "message": f"Human interaction response failed: {e}", "error_type": type(e).__name__}


    # --- Context Preparation ---
    def _prepare_context_for_llm(self, purpose: str, max_tokens: int = 4096):
        """Prepares the dynamic context string for LLM prompts."""
        logger.debug(f"Preparing context for purpose: {purpose}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        
        # Use the WorkingMemory's format_for_prompt method with appropriate sections
        if purpose == "state_evaluation":
            # For evaluation, include all relevant sections
            sections_to_include = [
                MemorySection.ACTIVE_GOAL,
                MemorySection.RECENT_ACTIONS,
                MemorySection.CONTEXT_ITEMS,
                MemorySection.RESOURCE_STATUS,
                MemorySection.LTM_RETRIEVALS,
                MemorySection.TOOL_OUTPUTS,
                MemorySection.STATE_FLAGS
            ]
        elif purpose == "action_selection":
            # For action selection, include all except maybe pending synthesis
            sections_to_include = [
                MemorySection.ACTIVE_GOAL,
                MemorySection.RECENT_ACTIONS,
                MemorySection.CONTEXT_ITEMS,
                MemorySection.LTM_RETRIEVALS,
                MemorySection.REASONING,
                MemorySection.TOOL_OUTPUTS,
                MemorySection.STATE_FLAGS
            ]
        elif purpose == "human_interaction":
            # For human interaction, prioritize goals and recent context
            sections_to_include = [
                MemorySection.ACTIVE_GOAL,
                MemorySection.CONTEXT_ITEMS,
                MemorySection.REASONING,
                MemorySection.LTM_RETRIEVALS
            ]
        else:
            # Default to all sections
            sections_to_include = [section for section in MemorySection]
            
        formatted_context = self.state.working_memory.format_for_prompt(sections_to_include)
        
        # Add LTM Retrieval if not already included
        if purpose != "action_selection" and purpose != "state_evaluation":
            # Query LTM based on the primary active goal or specific context needs
            ltm_results_str = ""
            active_goals = self.state.get_active_goals()
            if active_goals:
                primary_goal_desc = active_goals[0].description
                try:
                    # Example: Simple vector query based on goal description
                    logger.debug(f"Querying LTM (vector) based on goal: {primary_goal_desc[:50]}...", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                    vector_results = self.ltm.query_vector(query_embedding=primary_goal_desc, top_k=3)
                    if vector_results:
                        ltm_results_str += "### Relevant LTM Insights (Vector Search):\n"
                        for item in vector_results:
                            ltm_results_str += f"- (Score: {item.get('score', '?'):.2f}) {item.get('text', 'N/A')[:200]}...\n"

                    # Example: Could also add graph queries if relevant entities identified
                    # graph_results = self.ltm.query_graph(...)
                    # ltm_results_str += f"### Relevant LTM Connections (Graph Search):\n{json.dumps(graph_results)}\n"

                except Exception as e:
                    logger.error(f"Failed to query LTM during context preparation: {e}", exc_info=True, extra={"session_id": self.session_id, "cycle": self.cycle_count})
                    ltm_results_str += "[LTM Query Failed]\n"

            if ltm_results_str:
                formatted_context += f"\n## Long-Term Memory Retrieval:\n{ltm_results_str}"
        
        # Add resource state for evaluations if not already included
        if purpose == "state_evaluation" and MemorySection.RESOURCE_STATUS not in sections_to_include:
            res_dict = self.state.resources.dict()
            limits = self.resource_limits
            resource_text = f"""## Resource Status:
- Cost: ${res_dict['total_cost_usd']:.4f} / ${limits['total_budget_usd']:.2f}
- Tokens: {res_dict['total_tokens']} / {limits['max_total_tokens']}
- Cycles: {res_dict['cycle_count']} / {limits['max_cycles']}
- Elapsed Time: {res_dict['elapsed_time']:.2f}s
"""
            formatted_context += "\n\n" + resource_text
            
        logger.debug(f"Prepared context with {len(formatted_context)} chars", 
                    extra={"session_id": self.session_id, "cycle": self.cycle_count})
        
        return formatted_context

    # --- Prompt Generation ---
    
    def _create_state_evaluation_prompt(self, context: str) -> str:
        """Creates a prompt for state evaluation."""
        return f"""
# State Evaluation Task (Cycle: {self.cycle_count})

## Current Context & History
{context}

## Instructions
Analyze the provided context, history, goals, and resource status.
Evaluate the progress towards the most important active goal(s).
Identify any immediate blockers, risks, or opportunities.
Determine what kind of action is most needed next (e.g., gather info, reason, manage goals, synthesize memory).

Respond ONLY with a valid JSON object containing your evaluation:
```json
{{
    "goal_assessment": {{
        "primary_goal_id": "ID of the main goal being assessed or null",
        "progress_estimate": 0.0, // 0.0 to 1.0
        "blockers": ["List of blocking factors, if any"],
        "confidence": 0.8 // Confidence in assessment (0.0 to 1.0)
    }},
    "context_sufficiency": "sufficient", // "sufficient", "insufficient", "excessive"
    "critical_info_needed": "Specific info required, if context is insufficient",
    "suggested_next_action_type": "tool_call", // Suggest one ActionType value (e.g., llm_call, tool_call, memory_query, goal_management, memory_synthesis)
    "reasoning": "Brief justification for the suggested action type."
}}
```
"""

    def _create_action_selection_prompt(self, context: str, evaluation: Dict[str, Any]) -> str:
        """Creates a prompt for action selection."""
        return f"""
# Action Selection Task (Cycle: {self.cycle_count})

## Current Context & History
{context}

## State Evaluation Summary
```json
{json.dumps(evaluation, indent=2)}
```

## Available Action Types
{json.dumps([e.value for e in ActionType if e not in [ActionType.AWAIT_OVERSIGHT, ActionType.RESOURCE_CHECK]])}

## Instructions
Based on the context and the state evaluation, select the single best concrete action to take next.
Define the action type and necessary parameters.
If calling a tool, specify `tool_name` and required `params`.
If querying memory, specify `query_type` ('vector', 'graph', 'hybrid', 'metadata', 'emotional'), `query`, and optional `params` (like `top_k`).
If calling the LLM, specify `model` ('core' or 'cu'), `prompt`, optional `generation_config` overrides, and `use_cache` (boolean, typically true for 'core').
If managing goals, specify `operation` ('add', 'complete', 'fail', 'update_priority', 'decompose') and `goal_data`.
If synthesizing memory, use `memory_synthesis` type with minimal params.

Respond ONLY with a valid JSON object describing the action:
```json
{{
    "action_type": "tool_call",
    "description": "Concise description of the action's purpose.",
    "params": {{
        "tool_name": "search",
        "query": "Specific query for the tool"
        // Add other necessary params based on action_type
    }},
    "reasoning": "Brief justification for selecting this specific action and its parameters."
}}
```
"""

    # --- Oversight Handling ---
    def _requires_oversight(self, action: Action) -> bool:
        """Determines if an action needs human approval based on config and action type/params."""
        # Configurable list of sensitive tool names
        sensitive_tools = self.config.get("oversight", {}).get("sensitive_tools", ["twitter", "email", "database_write", "api_post", "financial_transaction"])
        # Configurable keywords in description
        sensitive_keywords = self.config.get("oversight", {}).get("sensitive_keywords", ["delete", "purchase", "transfer", "approve", "grant access"])
        # Flag to require oversight for all tool calls (for high caution)
        oversight_all_tools = self.config.get("oversight", {}).get("require_for_all_tools", False)
        # Flag to require oversight for core LLM calls (potentially expensive/powerful)
        oversight_core_llm = self.config.get("oversight", {}).get("require_for_core_llm", False)

        if action.type == ActionType.TOOL_CALL:
            if oversight_all_tools: return True
            tool_name = action.params.get("tool_name", "").lower()
            if tool_name in sensitive_tools: return True

        if action.type == ActionType.LLM_CALL:
             if oversight_core_llm and action.params.get("model") == "core": return True

        # Check keywords in description
        desc = action.description.lower()
        for keyword in sensitive_keywords:
            if keyword in desc: return True

        # Add other conditions as needed (e.g., high estimated cost?)

        return False # Default to no oversight needed

    def _request_oversight(self, action: Action):
        """Flags an action as needing oversight and logs it for external approval."""
        action.status = "awaiting_oversight"
        self.state.status = "awaiting_oversight" # Pause the agent's main loop progression
        self.state.current_action = action # Ensure the pending action is tracked

        logger.warning(f"Requesting oversight for Action ID: {action.id}", extra={
            "session_id": self.session_id, "cycle": self.cycle_count, "action_id": action.id,
            "action_type": action.type.value, "action_description": action.description, "action_params": action.params
        })

        # Log to Firestore for external monitoring/approval interface
        try:
            oversight_doc_ref = self.firestore_client.collection(self.oversight_collection_name).document(action.id)
            oversight_doc_ref.set({
                "session_id": self.session_id,
                "cycle_requested": self.cycle_count,
                "action_details": action.dict(),
                "status": "pending", # External system should update this to "approved" or "denied"
                "request_timestamp": firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
             logger.error(f"Failed to log oversight request to Firestore for action {action.id}: {e}", exc_info=True, extra={"session_id": self.session_id})
             # Critical failure? Should the agent stop? For now, log and continue in paused state.
             # This might mean the action can never be approved if Firestore fails.

    def _check_external_signals(self):
        """Checks Firestore for human input and oversight decisions."""
        # 1. Check for Oversight Decisions
        if self.state.status == "awaiting_oversight" and self.state.current_action:
            action_id = self.state.current_action.id
            try:
                oversight_doc_ref = self.firestore_client.collection(self.oversight_collection_name).document(action_id)
                snapshot = oversight_doc_ref.get()
                if snapshot.exists:
                    data = snapshot.to_dict()
                    oversight_status = data.get("status")
                    if oversight_status == "approved":
                         logger.info(f"Oversight approved for action {action_id}.", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                         self.state.current_action.status = "pending" # Ready for execution next cycle
                         self.state.status = "running" # Unpause agent
                    elif oversight_status == "denied":
                         logger.warning(f"Oversight denied for action {action_id}.", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                         self.state.current_action.status = "denied"
                         self.state.current_action.result = {"status": "denied", "message": "Action denied by human oversight."}
                         # Add context about denial and maybe trigger goal reassessment?
                         self.state.working_memory.add_context(f"OVERSIGHT_DENIED: Action '{self.state.current_action.description}' (ID: {action_id}) was denied.")
                         self.state.status = "running" # Unpause agent, will select new action
                         self.state.current_action = None # Clear the denied action
                    # Else: still pending, agent remains paused
                else:
                     logger.warning(f"Oversight document for action {action_id} not found.", extra={"session_id": self.session_id, "cycle": self.cycle_count})
                     # Remain paused? Or timeout?

            except Exception as e:
                 logger.error(f"Error checking oversight status for action {action_id}: {e}", exc_info=True, extra={"session_id": self.session_id})
                 # Remain paused to be safe

        # 2. Check for Human Input (only if not awaiting oversight)
        if self.state.status == "running":
             self._check_for_human_input()


    # --- Human-in-the-Loop (HITL) Implementation ---
    def _check_for_human_input(self) -> None:
        """Checks Firestore for unprocessed human input and injects it as a high-priority action."""
        try:
            query = self.firestore_client.collection(self.human_input_collection_name)\
                .where("session_id", "==", self.session_id)\
                .where("processed", "==", False)\
                .order_by("timestamp", direction=firestore.Query.ASCENDING)\
                .limit(1) # Process one at a time

            results = list(query.stream())
            if results:
                input_doc = results[0]
                input_data = input_doc.to_dict()
                input_id = input_doc.id
                input_text = input_data.get("text", "")
                input_timestamp = input_data.get("timestamp", datetime.now().isoformat())

                logger.info(f"Human input detected (ID: {input_id}). Injecting processing action.", extra={"session_id": self.session_id, "cycle": self.cycle_count})

                # Mark as processing immediately to prevent reprocessing by concurrent cycles/instances
                try:
                    input_doc.reference.update({"processing": True, "processing_start_time": firestore.SERVER_TIMESTAMP})
                except Exception as update_err:
                     logger.error(f"Failed to mark human input {input_id} as processing: {update_err}", exc_info=True)
                     return # Don't proceed if we can't mark it

                # Add human input to working memory
                self.state.working_memory.add_observation({
                    "summary": f"Human input: {input_text[:100]}...",
                    "timestamp": input_timestamp,
                    "type": "human_input",
                    "input_id": input_id,
                    "input_text": input_text,
                })

                # Add a high-priority goal for processing this input
                hitl_goal = Goal(
                     description=f"Process and respond to Human Input: {input_text[:100]}...",
                     priority=1.0, # Highest priority
                     context_pointers=[f"human_input:{input_id}"] # Link goal to input
                )
                
                # Store the input details in the goal's context for later retrieval
                hitl_goal.context_pointers.append(f"human_input_text:{input_text}")
                hitl_goal.context_pointers.append(f"human_input_id:{input_id}")
                
                # Add the goal to the state and update working memory
                self.state.goals.insert(0, hitl_goal) # Add to front
                self.state.working_memory.set_active_goal(hitl_goal.dict())

                # Mark as "queued" in Firestore, not "processing" yet
                input_doc.reference.update({
                    "status": "queued", 
                    "queued_cycle": self.cycle_count,
                    "goal_id": hitl_goal.id
                })

        except Exception as e:
            logger.error(f"Error checking for human input: {e}", exc_info=True, extra={"session_id": self.session_id})


    def _send_response_to_human(self, input_id: str, response: str) -> None:
        """Sends the agent's response back via Firestore."""
        logger.debug(f"Sending response for human input {input_id}", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        try:
            response_data = {
                "session_id": self.session_id,
                "input_id": input_id,
                "response_text": response,
                "response_timestamp": firestore.SERVER_TIMESTAMP,
                "cycle_responded": self.cycle_count,
            }
            # Add to agent response collection
            self.firestore_client.collection(self.agent_response_collection_name).add(response_data)

            # Update the original human input document to mark as processed
            human_input_query = self.firestore_client.collection(self.human_input_collection_name)\
                .where("session_id", "==", self.session_id)\
                .where("id", "==", input_id)
                
            human_input_docs = list(human_input_query.stream())
            if human_input_docs:
                human_input_docs[0].reference.update({
                    "processed": True,
                    "processing": False, # Ensure this is cleared
                    "processed_timestamp": firestore.SERVER_TIMESTAMP,
                    "status": "processed",
                })
                logger.info(f"Marked human input {input_id} as processed", extra={"session_id": self.session_id})
            else:
                logger.warning(f"Could not find human input document to mark as processed: {input_id}", 
                              extra={"session_id": self.session_id})

        except Exception as e:
            logger.error(f"Error sending response to human for input {input_id}: {e}", exc_info=True, extra={"session_id": self.session_id})
            # Don't crash the agent, just log the failure

    # --- Utility Methods ---
    def _parse_llm_json_response(self, response_text: str, default_if_error: Optional[Any] = None) -> Any:
        """Attempts to parse JSON from LLM response, handling markdown code blocks."""
        try:
            # Find JSON block ```json ... ```
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Assume the whole text is JSON if no block found (less reliable)
                json_str = response_text.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}. Raw text: '{response_text[:200]}...'",
                           extra={"session_id": self.session_id, "cycle": self.cycle_count})
            return default_if_error
        except Exception as e: # Catch other potential errors like regex issues
             logger.error(f"Unexpected error parsing LLM JSON response: {e}", exc_info=True,
                          extra={"session_id": self.session_id, "cycle": self.cycle_count})
             return default_if_error

    def _update_resource_usage(self, cost_usd: float = 0.0, input_tokens: int = 0, output_tokens: int = 0):
        """Updates the central resource tracking state."""
        self.state.resources.total_cost_usd += cost_usd
        self.state.resources.total_input_tokens += input_tokens
        self.state.resources.total_output_tokens += output_tokens
        
        # Log detailed usage if cost is significant
        if cost_usd > 0.01:
            logger.info(f"Significant resource usage: ${cost_usd:.4f}, {input_tokens + output_tokens} total tokens", 
                       extra={"session_id": self.session_id, "cycle": self.cycle_count})

    def _calculate_resource_usage(self, source_type: str, response: Any) -> Dict:
        """Calculates cost and tokens based on source and response metadata."""
        # Initialize with defaults
        cost = 0.0
        input_tokens = 0
        output_tokens = 0

        # LLM API calls
        if source_type.startswith("llm") or source_type == "cu_llm" or source_type == "core_llm":
            # Access usage metadata if available
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                
                # Calculate cost based on model type and pricing config
                if source_type == "cu_llm" or "flash" in source_type:
                    # Flash pricing - typically lower
                    input_cost_per_k = self.config.get("pricing", {}).get("flash_input_per_k", 0.0005)
                    output_cost_per_k = self.config.get("pricing", {}).get("flash_output_per_k", 0.0015)
                else:
                    # Pro pricing - typically higher
                    input_cost_per_k = self.config.get("pricing", {}).get("pro_input_per_k", 0.0010)
                    output_cost_per_k = self.config.get("pricing", {}).get("pro_output_per_k", 0.0030)
                
                # Calculate total cost
                cost += (input_tokens / 1000.0) * input_cost_per_k 
                cost += (output_tokens / 1000.0) * output_cost_per_k
            else:
                # Fallback estimation if no metadata available
                text_len = len(getattr(response, 'text', ''))
                prompt_len = len(getattr(response, '_raw_response', {}).get('prompt', ''))
                
                # Rough token estimation (characters / 4)
                input_tokens = prompt_len // 4
                output_tokens = text_len // 4
                
                # Apply a default minimal cost
                cost += 0.001 * (1 + (input_tokens + output_tokens) / 1000)

        # Tool usage costs
        elif source_type.startswith("tool_"):
            if isinstance(response, dict):
                # Some tools might report their cost directly
                cost += response.get("cost", 0.0)
                
                # Specific tool cost calculations
                if "google_search" in source_type:
                    # Google Search API has per-query costs
                    cost += self.config.get("pricing", {}).get("search_api_per_query", 0.005)
                elif "twitter" in source_type:
                    # Twitter API costs depend on tier and operation
                    cost += self.config.get("pricing", {}).get("twitter_api_per_call", 0.0025)

        # Log the calculated usage
        logger.debug(f"Resource usage for {source_type}: Cost=${cost:.6f}, InTokens={input_tokens}, OutTokens={output_tokens}",
                    extra={"session_id": self.session_id, "cycle": self.cycle_count})

        return {"cost_usd": cost, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def _update_resource_usage_from_response(self, source_type: str, response: Any):
        """Helper to calculate and update global state from a response."""
        usage = self._calculate_resource_usage(source_type, response)
        self._update_resource_usage(**usage)


    # --- State Persistence ---
    def _checkpoint_state(self, reason: str = "periodic") -> Optional[str]:
        """Saves the current agent state to GCS and updates Firestore."""
        checkpoint_start_time = time.time()
        checkpoint_id = f"{self.session_id}_{self.cycle_count}_{reason}_{int(checkpoint_start_time)}"
        logger.info(f"Creating checkpoint: {checkpoint_id}", extra={"session_id": self.session_id, "cycle": self.cycle_count, "reason": reason})

        try:
            # Serialize the *full* state for potential recovery
            # Be mindful of size, especially WM context and goals with subgoals
            state_data = self.state.dict()
            state_json = json.dumps(state_data, indent=2)

            # Save to GCS
            blob_name = f"checkpoints/{self.session_id}/{checkpoint_id}.json"
            blob = self.checkpoint_bucket.blob(blob_name)
            blob.upload_from_string(state_json, content_type="application/json")

            # Update latest state pointer in Firestore (for monitoring/quick view)
            # Store only essential summary, not the full state blob again
            firestore_summary = {
                "latest_checkpoint_id": checkpoint_id,
                "latest_checkpoint_reason": reason,
                "latest_checkpoint_timestamp": firestore.SERVER_TIMESTAMP,
                "cycle_count": self.state.resources.cycle_count,
                "status": self.state.status,
                "total_cost_usd": self.state.resources.total_cost_usd,
                "active_goals_count": len(self.state.get_active_goals()),
                # Maybe add primary goal description?
            }
            self.state_collection.document(self.session_id).set(firestore_summary, merge=True)

            duration = time.time() - checkpoint_start_time
            logger.info(f"Checkpoint {checkpoint_id} created successfully in {duration:.2f}s.",
                       extra={"session_id": self.session_id, "cycle": self.cycle_count, "checkpoint_id": checkpoint_id, "duration": duration})
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}", exc_info=True,
                        extra={"session_id": self.session_id, "cycle": self.cycle_count, "reason": reason})
            return None


    # --- Error Handling ---
    def _handle_general_error(self, error: Exception):
        """Handles unexpected errors during the cycle, attempts recovery goal."""
        error_type = type(error).__name__
        error_message = str(error)
        logger.error(f"Handling general cycle error: {error_type}: {error_message}",
                    extra={"session_id": self.session_id, "cycle": self.cycle_count, "error_type": error_type})

        # Add error details to working memory
        self.state.working_memory.add_context(f"ERROR ENCOUNTERED (Cycle {self.cycle_count}): {error_type} - {error_message}\n{traceback.format_exc(limit=3)}") # Add brief traceback

        # Create/prioritize a recovery goal if feasible
        if not any(g.description.startswith("Recover from error:") for g in self.state.get_active_goals()):
            recovery_goal = Goal(
                description=f"Recover from error: {error_type} encountered in cycle {self.cycle_count}",
                priority=1.0, # Make recovery top priority
                status="active"
            )
            self.state.goals.insert(0, recovery_goal) # Add to front
            # Update working memory with the new active goal
            self.state.working_memory.set_active_goal(recovery_goal.dict())
            logger.info("Added high-priority error recovery goal.", extra={"session_id": self.session_id, "cycle": self.cycle_count})
        else:
             logger.warning("Error recovery goal already exists.", extra={"session_id": self.session_id, "cycle": self.cycle_count})

        # Attempt a checkpoint after error
        self._checkpoint_state(f"error_handling_{error_type}")

    def _handle_termination(self, error: ResourceExceededError):
        """Handles graceful shutdown due to resource limits."""
        logger.critical(f"TERMINATION triggered due to resource limit: {error}",
                       extra={"session_id": self.session_id, "cycle": self.cycle_count})
        self.state.status = "terminated_resource_limit"

        # Try to perform final actions:
        # 1. Final Checkpoint
        logger.info("Attempting final checkpoint before termination...", extra={"session_id": self.session_id})
        self._checkpoint_state("termination")

        # 2. Log final state summary
        logger.info(f"Final State Summary: Cycles={self.cycle_count}, Cost=${self.state.resources.total_cost_usd:.4f}", 
                   extra={"session_id": self.session_id})

        # 3. Update Firestore status
        try:
            self.state_collection.document(self.session_id).set({
                 "status": "terminated_resource_limit",
                 "termination_reason": str(error),
                 "termination_timestamp": firestore.SERVER_TIMESTAMP,
                 "final_cycle_count": self.state.resources.cycle_count,
                 "final_cost_usd": self.state.resources.total_cost_usd,
            }, merge=True)
        except Exception as fs_err:
             logger.error(f"Failed to update final Firestore status during termination: {fs_err}", exc_info=True, extra={"session_id": self.session_id})

        # No further actions should be taken after this point.

class WorkingMemory:
    """Stores transient data for the current operational context."""
    def __init__(self):
        self.current_context: List[str] = [] # Holds recent LLM thoughts, tool results, LTM snippets
        self.action_history: List[Action] = [] # History of actions taken in recent cycles
        self.pending_data_for_synthesis: List[Tuple[Any, str, Dict]] = [] # Data, source, metadata
        self.parent_control_unit = None  # Will be set by the control unit after initialization

    def add_context(self, item: str, max_items: int = 20):
        self.current_context.append(str(item))
        # Basic recency trimming
        self.current_context = self.current_context[-max_items:]

    def add_action(self, action: 'Action', max_items: int = 10):
        self.action_history.append(action)
        self.action_history = self.action_history[-max_items:]

    def add_data_for_synthesis(self, data: Any, source: str, metadata: Dict):
        self.pending_data_for_synthesis.append((data, source, metadata))

    def clear_pending_synthesis(self):
        self.pending_data_for_synthesis = []

    def get_context_string(self) -> str:
        # Simple concatenation, could be more sophisticated (summarization?)
        return "\n".join(self.current_context)

    def get_action_history_string(self) -> str:
        return "\n".join([f"- {a.type.value}: {a.description}" for a in self.action_history])

    def dict(self): # For checkpointing
        return {
            "current_context": self.current_context,
            "action_history": [a.dict() for a in self.action_history],
            # Note: pending_data_for_synthesis is transient, likely not checkpointed directly
        }
        
    def format_for_prompt(self, sections) -> str:
        """Formats working memory content for inclusion in an LLM prompt.
        
        Args:
            sections: List of MemorySection enums to include
            
        Returns:
            Formatted string with the requested sections
        """
        formatted_sections = []
        
        # Format active goals if requested
        if MemorySection.ACTIVE_GOAL in sections and hasattr(self, 'parent_control_unit'):
            active_goals = self.parent_control_unit.state.get_active_goals()
            if active_goals:
                goals_section = "## Current Active Goals (Priority Order):\n"
                for i, goal in enumerate(active_goals[:5]):  # Limit shown goals
                    goals_section += f"{i+1}. (ID: {goal.id}) {goal.description}\n"
                formatted_sections.append(goals_section)
        
        # Format recent actions if requested
        if MemorySection.RECENT_ACTIONS in sections:
            if self.action_history:
                actions_section = "## Recent Actions Taken:\n"
                actions_section += "\n".join([f"- {a.type.value}: {a.description}" for a in self.action_history[-5:]])
                formatted_sections.append(actions_section)
        
        # Format working memory context items if requested
        if MemorySection.CONTEXT_ITEMS in sections:
            if self.current_context:
                context_section = "## Recent Context Items:\n"
                context_section += "\n".join(self.current_context[-10:])  # Last 10 items
                formatted_sections.append(context_section)

        # Format LTM retrievals if requested
        if MemorySection.LTM_RETRIEVALS in sections and hasattr(self, 'ltm_retrievals'):
            if self.ltm_retrievals:
                ltm_section = "## Recent LTM Retrievals:\n"
                # Format based on expected retrieval structure
                for retrieval in self.ltm_retrievals[-3:]:  # Last 3 retrievals
                    ltm_section += f"- {retrieval.get('text', 'No text')} (Score: {retrieval.get('score', '?'):.2f})\n"
                formatted_sections.append(ltm_section)
                
        # Format reasoning if requested
        if MemorySection.REASONING in sections and hasattr(self, 'reasoning_steps'):
            if self.reasoning_steps:
                reasoning_section = "## Recent Reasoning Steps:\n"
                reasoning_section += "\n".join(self.reasoning_steps[-3:])
                formatted_sections.append(reasoning_section)
                
        # Format tool outputs if requested
        if MemorySection.TOOL_OUTPUTS in sections and hasattr(self, 'tool_outputs'):
            if self.tool_outputs:
                tools_section = "## Recent Tool Outputs:\n"
                for tool_output in self.tool_outputs[-3:]:
                    tools_section += f"- {tool_output.get('tool')}: {tool_output.get('result', '')[:100]}...\n"
                formatted_sections.append(tools_section)
                
        # Format state flags if requested
        if MemorySection.STATE_FLAGS in sections and hasattr(self, 'state_flags'):
            if self.state_flags:
                flags_section = "## Current State Flags:\n"
                for flag, value in self.state_flags.items():
                    flags_section += f"- {flag}: {value}\n"
                formatted_sections.append(flags_section)
        
        # Format resource status if requested
        if MemorySection.RESOURCE_STATUS in sections and hasattr(self, 'parent_control_unit'):
            res_dict = self.parent_control_unit.state.resources.dict()
            limits = self.parent_control_unit.resource_limits
            resource_section = f"""## Resource Status:
- Cost: ${res_dict['total_cost_usd']:.4f} / ${limits['total_budget_usd']:.2f}
- Tokens: {res_dict['total_tokens']} / {limits['max_total_tokens']}
- Cycles: {res_dict['cycle_count']} / {limits['max_cycles']}
- Elapsed Time: {res_dict['elapsed_time']:.2f}s
"""
            formatted_sections.append(resource_section)
        
        # Combine all sections
        return "\n\n".join(formatted_sections)