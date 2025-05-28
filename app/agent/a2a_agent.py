# File: app/agent/manus.py

# --- Keep your existing imports ---
from typing import Dict, List, Optional, TYPE_CHECKING, Any
import uuid
import asyncio
import httpx

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent # Keep this
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.delegate_task import ListAvailableAgentsTool, DelegateTaskToAgentTool
import json
# --- Import models for type hinting and use ---
if TYPE_CHECKING:
    from app.models import AgentCard, AgentCapability, A2AMessageRequest
# For actual runtime use of models (e.g. instantiation), do a direct import inside methods or ensure loaded
from app.schema import Message # <<<< ADD THIS IMPORT FOR CREATING MESSAGE OBJECTS


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 40

    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
            ListAvailableAgentsTool(),
            DelegateTaskToAgentTool(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None
    connected_servers: Dict[str, str] = Field(default_factory=dict)
    _initialized: bool = False


    async def _execute_tool_logic(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Executes specific logic for tools, especially those needing Manus instance context.
        """
        logger.debug(f"Manus ({self._manus_agent_id}) attempting to execute tool '{tool_name}' with args: {tool_args} via _execute_tool_logic.")
        # Assuming self.available_tools.get_tool(tool_name) correctly retrieves the tool instance
        tool_instance = self.available_tools.get_tool(tool_name) 
        
        if not tool_instance:
            logger.error(f"Tool '{tool_name}' not found in Manus's available tools.")
            return f"Error: Tool '{tool_name}' not found."

        # --- Specific handling for tools needing context ---
        if tool_name == "list_available_agents": # Match by the tool's 'name' attribute string
            # ListAvailableAgentsTool's execute method needs the agent's base URL.
            # The LLM provides no arguments for this tool (tool_args should be {}).
            logger.info(f"Calling ListAvailableAgentsTool.execute with agent_base_url='{self._a2a_base_url}'")
            return await tool_instance.execute(agent_base_url=self._a2a_base_url) # Pass context

        elif tool_name == "delegate_task_to_agent": # Match by the tool's 'name' attribute string
            # DelegateTaskToAgentTool's execute method needs the Manus instance itself
            # and arguments from the LLM (target_agent_id, task_prompt, etc.).
            logger.info(f"Calling DelegateTaskToAgentTool.execute with manus_agent_instance and LLM args: {tool_args}")
            return await tool_instance.execute(
                manus_agent_instance=self, # Pass context (the Manus agent instance itself)
                **tool_args  # Unpack LLM-provided args like target_agent_id, task_prompt
            )
        
        # --- Fallback for other tools that take arguments directly from LLM ---
        else:
            logger.info(f"Calling generic tool '{tool_name}' with args: {tool_args}")
            # Assuming other tools have an 'execute' method and take their args directly from tool_args
            if hasattr(tool_instance, 'execute') and callable(tool_instance.execute):
                return await tool_instance.execute(**tool_args)
            elif callable(tool_instance): # If tool itself is callable (less common for BaseTool pattern)
                return await tool_instance(**tool_args)
            else:
                logger.error(f"Tool '{tool_name}' is not callable or has no 'execute' method.")
                return f"Error: Tool '{tool_name}' cannot be executed or is not found."
    
        # Place this method inside your Manus class definition:
    # class Manus(ToolCallAgent):
    #     # ... other methods like __init__, create, think, _execute_tool_logic ...

    async def act(self) -> None: # The signature might slightly vary if your base class dictates it, but `self` is key.
        """
        Executes the tool calls selected by the LLM in the previous `think` step.
        This method calls `_execute_tool_logic` to handle context injection for specific tools.
        """
        print('Esto se ejecuta')
        # self.tool_calls should be populated by your LLM interaction logic,
        # typically after super().think() or a similar call in your Manus.think() method.
        # It's expected to be a list of tool call objects (e.g., from OpenAI, they have 'id', 'function.name', 'function.arguments').
        if not self.tool_calls:
            logger.debug(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: No tool calls to execute.")
            return

        logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Executing tool calls: {self.tool_calls}")
        
        # --- Ensure necessary imports are at the top of your agent file ---
        # import json
        # from app.schema import Message # Or your specific Message model for memory
        # from app.logger import logger # If logger is not already globally available

        for tool_call in self.tool_calls:
            # The structure of tool_call depends on your LLM provider (e.g., OpenAI).
            # These getattr calls provide a safe way to access nested attributes.
            tool_name = getattr(getattr(tool_call, 'function', None), 'name', None)
            tool_call_id = getattr(tool_call, 'id', None) # Essential for mapping response back to the call
            raw_args_str = getattr(getattr(tool_call, 'function', None), 'arguments', "{}")

            if not tool_name or not tool_call_id:
                logger.error(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Invalid tool_call object encountered: {tool_call}. Missing name or id.")
                # Optionally add an error message to memory here for this malformed tool_call
                # self.memory.add_message(Message.system_message(f"Error: Malformed tool call received: {tool_call}"))
                continue 
            
            tool_args = {} # Default to empty dict for arguments
            try:
                # Ensure raw_args_str is not None and not empty before trying to load.
                # Some LLMs might return an empty string or "null" for no arguments.
                if raw_args_str and raw_args_str.strip() and raw_args_str.lower() != "null":
                    tool_args = json.loads(raw_args_str)
                
                logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Preparing to execute tool: '{tool_name}' (Call ID: {tool_call_id}) with parsed args: {tool_args}")

                # Use our custom dispatcher (_execute_tool_logic) that knows about tool contexts
                observation_content = await self._execute_tool_logic(tool_name, tool_args)
                logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Tool '{tool_name}' (Call ID: {tool_call_id}) execution result (first 500 chars): {str(observation_content)[:500]}...")

            except json.JSONDecodeError as e_json:
                logger.error(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: JSONDecodeError for tool '{tool_name}' (Call ID: {tool_call_id}) with raw args '{raw_args_str}': {e_json}", exc_info=True)
                observation_content = f"Error: Invalid arguments provided for tool {tool_name}. Expected valid JSON. Details: {e_json}"
            except Exception as e_exec:
                logger.error(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Error during _execute_tool_logic for '{tool_name}' (Call ID: {tool_call_id}): {e_exec}", exc_info=True)
                observation_content = f"Error executing tool {tool_name}: {str(e_exec)}"
            
            # Add the observation (tool response) back to memory
            # This part is CRITICAL and depends heavily on your agent's memory and Message schema.
            if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'add_message'):
                try:
                    # Ensure content is a string. Some LLMs/frameworks might be picky.
                    stringified_content = str(observation_content)
                    
                    # Create a tool response message. The exact method and parameters will
                    # depend on your `app.schema.Message` class and how it expects tool responses.
                    # This example assumes a factory method `tool_response_message` on your Message class.
                    tool_response_msg = Message.tool_message(
                        content=stringified_content,
                        name=tool_name,  # Pass the name of the tool
                        tool_call_id=tool_call_id
                    )
                    self.memory.add_message(tool_response_msg)
                    logger.debug(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Added observation for tool '{tool_name}' (Call ID: {tool_call_id}) to memory.")
                except Exception as e_mem:
                    logger.error(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Failed to add tool observation to memory for '{tool_name}': {e_mem}", exc_info=True)
            else:
                logger.warning(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) - act: Memory object or 'add_message' method not found. Observation for '{tool_name}' not recorded.")

        self.tool_calls = [] # Clear processed tool calls


    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
        return self


    @classmethod
    async def create(cls, base_url: str, **kwargs: Any) -> "Manus":
        logger.debug(f"Manus.create: Entered. base_url='{base_url}', kwargs={kwargs}")
        try:
            instance = cls(**kwargs)
            logger.debug(f"Manus.create: Initial instance created via cls(**kwargs). Type: {type(instance)}, Object ID: {id(instance)}")

            # --- Debug Memory Initialization (Updated to check for 'add_message') ---
            logger.debug("Manus.create: Checking for 'memory' attribute post-instantiation...")
            if hasattr(instance, 'memory') and instance.memory is not None:
                logger.info(f"Manus.create: 'memory' attribute FOUND. Type: {type(instance.memory)}")
                if hasattr(instance.memory, 'add_message'): # <<< CHECK FOR 'add_message'
                    logger.info("Manus.create: instance.memory.add_message method FOUND.")
                else:
                    logger.warning("Manus.create: instance.memory.add_message method NOT FOUND!")
                    logger.debug(f"Manus.create: Available attributes/methods on instance.memory: {dir(instance.memory)}")
                if hasattr(instance.memory, 'messages'):
                    logger.debug(f"Manus.create: Initial memory messages (if any): {instance.memory.messages}")
                else:
                    logger.debug("Manus.create: instance.memory does not have a 'messages' attribute.")
            else:
                logger.warning("Manus.create: 'memory' attribute NOT FOUND or is None on instance!")
                logger.debug(f"Manus.create: All attributes on instance: {dir(instance)}")
            # --- End Debug Memory Initialization ---

        except Exception as e_init:
            logger.error(f"Manus.create: CRITICAL - Exception during cls(**kwargs) Pydantic initialization: {e_init}", exc_info=True)
            raise RuntimeError(f"Failed during Pydantic model initialization in Manus.create: {e_init}") from e_init

        current_step = "setting_a2a_attributes"
        try:
            instance._manus_agent_id = str(uuid.uuid4())
            agent_base_name = getattr(instance, 'name', 'DefaultManusName')
            if not isinstance(agent_base_name, str): agent_base_name = "DefaultManusName"
            instance._manus_public_name = f"{agent_base_name}-{instance._manus_agent_id[:8]}"
            instance._a2a_base_url = base_url
            instance._a2a_full_endpoint = f"{base_url}/a2a/{instance._manus_agent_id}/message"
            logger.debug(f"Manus.create: A2A attributes set for {instance._manus_public_name}")

        except Exception as e_attrs:
            logger.error(f"Manus.create: CRITICAL - Error during A2A attribute assignment (at step: '{current_step}'): {e_attrs}", exc_info=True)
            raise RuntimeError(f"Failed to set critical A2A attributes on Manus instance (at step: '{current_step}'): {e_attrs}") from e_attrs

        logger.info(f"Manus instance '{instance._manus_public_name}' ({instance._manus_agent_id}) created.")
        logger.info(f"A2A Endpoint will be: {instance._a2a_full_endpoint}")

        try:
            await instance.initialize_mcp_servers()
            instance._initialized = True
            logger.debug(f"Manus.create: MCP servers initialized for {instance._manus_agent_id}")
        except Exception as e_mcp:
            logger.error(f"Manus.create: Error initializing MCP servers for {instance._manus_agent_id}: {e_mcp}", exc_info=True)
            instance._initialized = False

        return instance

    async def get_agent_card(self) -> 'AgentCard':
        from app.models import AgentCard, AgentCapability
        logger.debug(f"Manus.get_agent_card: Generating agent card for {getattr(self, '_manus_public_name', 'N/A')}")
        agent_capabilities = [
            AgentCapability(name="general_task_solver", description=self.description),
            AgentCapability(name="execute_prompt", description="Can process a textual prompt to perform actions using available tools."),
        ]
        logger.debug(f"Manus.get_agent_card: Accessing tools from self.available_tools (type: {type(self.available_tools)}).")
        if hasattr(self.available_tools, 'tools') and self.available_tools.tools is not None:
            logger.debug(f"Manus.get_agent_card: self.available_tools.tools found (type: {type(self.available_tools.tools)}). Iterating...")
            try:
                for tool_instance in self.available_tools.tools:
                    tool_name = getattr(tool_instance, 'name', None)
                    tool_description_attr = getattr(tool_instance, 'description', 'No specific description available.')
                    tool_description = str(tool_description_attr) if tool_description_attr is not None else 'Description not available.'
                    if tool_name:
                        logger.debug(f"Manus.get_agent_card: Adding tool capability: {tool_name}")
                        agent_capabilities.append(
                            AgentCapability(name=f"tool:{tool_name}", description=tool_description)
                        )
                    else:
                        logger.warning(f"Manus.get_agent_card: Found a tool instance of type {type(tool_instance)} without a 'name' attribute. Skipping for capabilities.")
            except TypeError as te:
                logger.error(f"Manus.get_agent_card: TypeError while iterating self.available_tools.tools: {te}. Tools might not be a list or iterable.", exc_info=True)
            except Exception as e_tool_iter:
                logger.error(f"Manus.get_agent_card: Unexpected error while iterating tools: {e_tool_iter}", exc_info=True)
        else:
            logger.warning("Manus.get_agent_card: self.available_tools does not have a 'tools' attribute or it's None. Skipping tool capabilities.")
        card_description = f"A {getattr(self, 'name', 'Manus')} agent. {getattr(self, 'description', 'No specific description.')}"
        return AgentCard(
            agent_id=getattr(self, '_manus_agent_id', 'UNKNOWN_AGENT_ID'),
            agent_name=getattr(self, '_manus_public_name', 'Unknown Manus Agent'),
            description=card_description,
            capabilities=agent_capabilities,
            a2a_endpoint=getattr(self, '_a2a_full_endpoint', 'UNKNOWN_A2A_ENDPOINT')
        )

    async def handle_a2a_message(self, request: 'A2AMessageRequest') -> Dict[str, any]:
        from app.models import A2AMessageRequest
        logger.info(f"Manus ({self._manus_agent_id}) received A2A message from {request.sender_agent_id} of type {request.message_type}")
        if request.message_type == "execute_capability":
            capability_name = request.payload.get("capability_name")
            params = request.payload.get("params", {})
            if capability_name == "execute_prompt" or capability_name == "general_task_solver":
                prompt_to_run = params.get("prompt")
                if not isinstance(prompt_to_run, str) or not prompt_to_run.strip():
                    return {"status": "error", "message": "Missing or invalid 'prompt' in payload for capability"}
                try:
                    logger.info(f"Manus ({self._manus_agent_id}) executing A2A prompt: '{prompt_to_run}'")
                    result = await self.run(prompt=prompt_to_run)
                    return {"status": "success", "data": result if result is not None else "Task initiated/processed"}
                except Exception as e:
                    logger.error(f"Error running capability '{capability_name}' via A2A for agent {self._manus_agent_id}: {e}", exc_info=True)
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": f"Unknown or unsupported capability: {capability_name}"}
        else:
            return {"status": "error", "message": f"Unsupported A2A message type: {request.message_type}"}

    async def send_a2a_message_to_another_agent(self, target_agent_card: 'AgentCard', message_type: str, payload: dict) -> Optional[Dict[str, Any]]:
        from app.models import A2AMessageRequest, AgentCard
        if not target_agent_card.a2a_endpoint:
            logger.error(f"Target agent {target_agent_card.agent_id} has no A2A endpoint defined in its card.")
            return None
        message_to_send = A2AMessageRequest(
            sender_agent_id=self._manus_agent_id,
            message_type=message_type,
            payload=payload
        )
        logger.info(f"Manus ({self._manus_agent_id}) sending A2A to {target_agent_card.agent_id} ({target_agent_card.a2a_endpoint}) type '{message_type}'")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(target_agent_card.a2a_endpoint, json=message_to_send.model_dump())
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error sending A2A from {self._manus_agent_id} to {target_agent_card.agent_id}: {e.response.status_code} - {e.response.text}", exc_info=True)
            except httpx.RequestError as e:
                logger.error(f"Request error sending A2A from {self._manus_agent_id} to {target_agent_card.agent_id}: {e}", exc_info=True)
            return None

    async def initialize_mcp_servers(self) -> None:
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(f"Connected to MCP server {server_id} at {server_config.url}")
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(f"Connected to MCP server {server_id} using command {server_config.command}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: Optional[List[str]] = None,
    ) -> None:
        if use_stdio:
            await self.mcp_clients.connect_stdio(server_url, stdio_args or [], server_id)
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url
        new_tools = [tool for tool in self.mcp_clients.tools if tool.server_id == server_id]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            all_server_ids = list(self.mcp_clients.clients.keys())
            for s_id in all_server_ids:
                 await self.mcp_clients.disconnect(s_id)
            self.connected_servers.clear()
        base_tools = [
            tool for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool) or tool.server_id not in self.mcp_clients.clients
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        if self._initialized:
            logger.info(f"Manus ({self._manus_agent_id}) cleaning up MCP connections.")
            await self.disconnect_mcp_server()
            self._initialized = False
            logger.info(f"Manus ({self._manus_agent_id}) MCP cleanup complete.")

    async def think(self) -> bool:
        if not self._initialized:
            logger.warning(f"Manus ({self._manus_agent_id}) was not initialized. Attempting to initialize MCP servers before thinking.")
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        # Ensure self.memory and self.memory.messages exist before accessing
        recent_messages = []
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'messages') and self.memory.messages:
            recent_messages = self.memory.messages[-3:]
        else:
            logger.warning(f"Manus.think: Memory or memory.messages not available for agent {getattr(self, '_manus_agent_id', 'N/A')}")

        browser_in_use = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls and
            any(tc.function.name == BrowserUseTool().name for tc in msg.tool_calls)
            for msg in recent_messages
        )

        if browser_in_use:
            self.next_step_prompt = await self.browser_context_helper.format_next_step_prompt()

        result = await super().think() # This will use self.memory.add_message correctly
        self.next_step_prompt = original_prompt
        return result

    async def run(self, prompt: str, **kwargs: Any) -> Optional[Dict[str, any]]:
        logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) 'run' method called with prompt: '{prompt}'")

        if not self._initialized:
            logger.warning("Manus not initialized in run method. Attempting to initialize.")
            await self.initialize_mcp_servers()
            self._initialized = True

        # 1. Add user prompt to memory
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'add_message'):
            try:
                user_prompt_message = Message.user_message(content=prompt)
                self.memory.add_message(user_prompt_message)
                logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) added user prompt to memory via add_message.")
            except Exception as e_mem_add:
                logger.error(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) failed to add prompt to memory: {e_mem_add}", exc_info=True)
        elif hasattr(self, 'memory') and self.memory:
             logger.warning(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) agent memory FOUND, but 'add_message' method NOT available. Prompt not added. Methods: {dir(self.memory)}")
        else:
            logger.warning(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) agent memory NOT FOUND. Prompt not added to memory.")

        # 2. Run the think/step loop
        max_run_steps = kwargs.get("max_run_steps", self.max_steps // 2 or 5)
        for i in range(max_run_steps):
            logger.info(f"Run step {i+1}/{max_run_steps} for prompt: {prompt}")

            # This original check might still be useful if ToolCallAgent sometimes sets it,
            # but we won't rely on it as the primary mechanism for this solution.
            if hasattr(self, 'is_terminated') and self.is_terminated():
                 logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) terminated during run loop (via parent's is_terminated).")
                 break

            # Call self.think() (which internally calls super().think() from ToolCallAgent)
            # ToolCallAgent.think() is expected to populate self.tool_calls or a similar attribute.
            # It also returns a boolean indicating if the agent should continue.
            should_continue_thinking = await self.think()

            if not should_continue_thinking:
                logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) 'think' decided no further action or no tools selected.")
                break

            # MODIFICATION: Inspect self.tool_calls (or the attribute ToolCallAgent uses)
            # This attribute should be populated by ToolCallAgent's think() method.
            # The logs "🧰 Tools being prepared: ['terminate']" suggest such an attribute exists.
            terminate_tool_selected_this_step = False
            if hasattr(self, 'tool_calls') and self.tool_calls: # 'tool_calls' is a common attribute name
                for tool_call_item in self.tool_calls:
                    # The structure of tool_call_item needs to be determined.
                    # Common structures:
                    # 1. A Pydantic model: tool_call_item.function.name
                    # 2. A dictionary: tool_call_item.get('name') or tool_call_item.get('function', {}).get('name')
                    # Based on your logs for tool arguments like {"status":"success"},
                    # it's likely each item in self.tool_calls is an object or dict
                    # that contains the tool name and its arguments.

                    tool_name_in_call = None
                    if hasattr(tool_call_item, 'function') and hasattr(tool_call_item.function, 'name'): # Langchain like
                        tool_name_in_call = tool_call_item.function.name
                    elif isinstance(tool_call_item, dict) and 'name' in tool_call_item: # Simple dict
                         tool_name_in_call = tool_call_item['name']
                    # Add more checks here if the structure is different, e.g., some frameworks might just store a list of names.

                    if tool_name_in_call == Terminate().name: # Terminate().name is "terminate"
                        terminate_tool_selected_this_step = True
                        logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) run loop: Detected '{Terminate().name}' tool selected by 'think' method.")
                        break

            # Execute the selected tools using self.act()
            if hasattr(self, 'act'):
                await self.act() # act() executes tools from self.tool_calls and adds observations to memory
            else:
                logger.warning("Manus agent 'step' or 'act' method not found. Cannot execute tools.")
                break # Critical component missing

            # If the 'terminate' tool was selected in this step's 'think' phase, break the loop
            # after 'act' has had a chance to execute it (and any other tools selected for this step).
            if terminate_tool_selected_this_step:
                logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) run loop: Terminating because '{Terminate().name}' tool was executed this step.")
                break

        # 3. Formulate a response
        final_response_content = "Task processing initiated. Check agent logs or subsequent interactions for results."
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'messages') and self.memory.messages:
            for msg in reversed(self.memory.messages):
                if msg.role == "assistant" and msg.content:
                    final_response_content = msg.content
                    break

        logger.info(f"Manus ({getattr(self, '_manus_agent_id', 'N/A')}) finished 'run' method for prompt: '{prompt}'")
        return {"result": final_response_content, "agent_id": getattr(self, '_manus_agent_id', 'N/A')}
