# File: app/tool/discovery_and_delegation.py

from typing import List, Dict, Any, TYPE_CHECKING, Optional
import httpx
import json # For type hinting or potential direct use, though httpx handles response.json()

# Assuming BaseTool is in app.tool and you'll import it where this tool is defined
from app.tool import BaseTool # Replace 'app.tool' with the actual import path for BaseTool

# --- Models for Type Hinting ---
if TYPE_CHECKING:
    from app.models import AgentCard # Your Pydantic model for AgentCard
    from app.agent.a2a_agent import Manus # For type hinting manus_agent_instance


class ListAvailableAgentsTool(BaseTool):
    name: str = "list_available_agents"
    description: str = (
        "Fetches a list of all currently registered and available agents, "
        "including their IDs, names, descriptions, and capabilities. "
        "Use this to find out which other agents you can delegate tasks to and what they can do."
    )
    # This tool takes no arguments from the LLM.
    # The `agent_base_url` will be provided by the Manus agent during execution.
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {}, # No parameters for the LLM to provide
        "required": [],
    }

    async def execute(self, agent_base_url: str) -> List[Dict[str, Any]]:
        """
        Fetches registered agents from the discovery service.

        Args:
            agent_base_url: The base URL of the FastAPI server where this agent
                            (and its discovery service) is running. This is injected
                            by the calling Manus agent, not by the LLM.
        
        Returns:
            A list of dictionaries, where each dictionary summarizes an agent's card,
            or a list containing an error dictionary if an issue occurs.
        """
        if not agent_base_url:
            return [{"error": "agent_base_url was not provided to ListAvailableAgentsTool's execute method."}]

        discovery_url = f"{agent_base_url.rstrip('/')}/discovery/agents"
        # print(f"ListAvailableAgentsTool: Contacting discovery URL: {discovery_url}") # For debugging

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(discovery_url)
                response.raise_for_status()
                agents_data = response.json()

                summaries: List[Dict[str, Any]] = []
                for agent_details in agents_data:
                    try:
                        # Construct a summary. If app.models.AgentCard is available,
                        # you could validate agent_details against it first.
                        summary = {
                            "agent_id": agent_details.get("agent_id", "N/A"),
                            "agent_name": agent_details.get("agent_name", "N/A"),
                            "description": agent_details.get("description", "No description."),
                            "capabilities": [cap.get("name") for cap in agent_details.get("capabilities", []) if cap.get("name")],
                            "a2a_endpoint": agent_details.get("a2a_endpoint")
                        }
                        summaries.append(summary)
                    except Exception as e_parse: # Catch errors parsing individual agent cards
                        summaries.append({"error": f"Could not parse an agent card: {str(e_parse)}", "raw_details": agent_details})
                
                if not summaries and agents_data: # If parsing failed for all
                     return [{"warning": "Fetched agent data but failed to summarize it.", "raw_data": agents_data}]
                elif not agents_data:
                    return [{"info": "No agents are currently registered with the discovery service."}]
                return summaries

            except httpx.HTTPStatusError as e:
                return [{"error": f"HTTP error discovering agents: Status {e.response.status_code} - {e.response.text}", "url": discovery_url}]
            except httpx.RequestError as e:
                return [{"error": f"Request error discovering agents: {str(e)}", "url": discovery_url}]
            except json.JSONDecodeError as e: # httpx.Response.json() raises this if content-type is json but body is not valid json
                return [{"error": f"Failed to decode JSON response from discovery service: {str(e)}", "url": discovery_url}]
            except Exception as e: # Catch-all for other unexpected errors
                return [{"error": f"An unexpected error occurred in ListAvailableAgentsTool: {str(e)}", "url": discovery_url}]


class DelegateTaskToAgentTool(BaseTool):
    name: str = "delegate_task_to_agent"
    description: str = (
        "Delegates a specific task (prompt) to another agent identified by its ID. "
        "Use 'list_available_agents' first to find an appropriate agent and its ID. "
        "The task will be sent to the target agent for execution."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "target_agent_id": {
                "type": "string",
                "description": "The unique ID of the agent to delegate the task to. Use 'list_available_agents' to find this.",
            },
            "task_prompt": {
                "type": "string",
                "description": "The specific prompt or task description for the target agent.",
            },
            "capability_to_execute": {
                "type": "string",
                "description": "The capability the target agent should use (e.g., 'execute_prompt'). Default: 'execute_prompt'.",
            },
            "message_type": {
                "type": "string",
                "description": "The type of A2A message to send (usually 'execute_capability'). Default: 'execute_capability'.",
            },
        },
        "required": ["target_agent_id", "task_prompt"],
    }

    async def execute(
        self,
        target_agent_id: str,
        task_prompt: str,
        manus_agent_instance: 'Manus' ,
        capability_to_execute: str = "execute_prompt", 
        message_type: str = "execute_capability",
        # Contextual argument injected by Manus agent:
        
    ) -> Dict[str, Any]:
        """
        Delegates a task to a specified agent.

        Args:
            target_agent_id: The ID of the agent to delegate to (from LLM).
            task_prompt: The prompt/task for the target agent (from LLM).
            capability_to_execute: The capability to invoke on the target agent (from LLM, with default).
            message_type: The A2A message type (from LLM, with default).
            manus_agent_instance: The instance of the Manus agent calling this tool.
                                  Needed to access `_a2a_base_url` and `send_a2a_message_to_another_agent`.
                                  This is injected by the calling Manus agent.
        
        Returns:
            A dictionary containing the status of the delegation and the response
            from the target agent, or an error message.
        """
        # Ensure you have this import if you are using the AgentCard Pydantic model
        # It's placed here to be close to its usage and avoid top-level if not always needed.
        from app.models import AgentCard 

        if not manus_agent_instance: # This check is crucial
            return {"status": "error", "message": "Manus agent instance not provided to DelegateTaskToAgentTool's execute method."}
        if not hasattr(manus_agent_instance, "_a2a_base_url") or not hasattr(manus_agent_instance, "send_a2a_message_to_another_agent"):
            return {"status": "error", "message": "Provided Manus agent instance is missing required attributes for delegation (_a2a_base_url or send_a2a_message_to_another_agent)."}
         #To avoid registered itself
        if target_agent_id == manus_agent_instance._manus_agent_id:
            from app.logger import logger
            logger.warning(f"DelegateTaskToAgentTool: Attempted to delegate task to self ({target_agent_id}). This is not allowed.")
            return{
                "status":"error",
                "message":f"Delegate to self is not allowed. Please choose another agent. Agent ID: {target_agent_id}"
            }



        agent_base_url = manus_agent_instance._a2a_base_url
        target_card_url = f"{agent_base_url.rstrip('/')}/discovery/agents/{target_agent_id}"
        target_agent_card_obj: Optional[AgentCard] = None # Explicitly Optional

        # print(f"DelegateTaskToAgentTool: Fetching card for {target_agent_id} from {target_card_url}") # For debugging

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(target_card_url)
                response.raise_for_status()
                target_agent_card_data = response.json()
                # Validate data against the AgentCard Pydantic model
                target_agent_card_obj = AgentCard(**target_agent_card_data)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return {"status": "error", "message": f"Target agent with ID '{target_agent_id}' not found in discovery service via {target_card_url}."}
                return {"status": "error", "message": f"HTTP error fetching target agent card: Status {e.response.status_code} - {e.response.text}"}
            except httpx.RequestError as e: # Network errors
                return {"status": "error", "message": f"Request error fetching target agent card: {str(e)}"}
            except Exception as e: # Catches Pydantic ValidationError if AgentCard(**data) fails, or other unexpected errors
                return {"status": "error", "message": f"Failed to retrieve, parse, or validate agent card for '{target_agent_id}': {str(e)}"}

        if not target_agent_card_obj: # Should have been caught by exceptions, but as a safeguard
            return {"status": "error", "message": f"Could not obtain a valid agent card for target_agent_id '{target_agent_id}'."}

        # Prepare payload for the A2A message
        payload = {
            "capability_name": capability_to_execute,
            "params": {"prompt": task_prompt}
        }
        
        # print(f"DelegateTaskToAgentTool: Sending A2A to {target_agent_card_obj.agent_name} ({target_agent_card_obj.a2a_endpoint})") # For debugging

        try:
            # Use the Manus agent's own method to send the A2A message
            response_from_target = await manus_agent_instance.send_a2a_message_to_another_agent(
                target_agent_card=target_agent_card_obj,
                message_type=message_type, # e.g., "execute_capability"
                payload=payload
            )
 
            if response_from_target is not None:
                return {"status": "delegation_successful", "delegated_to_agent_id": target_agent_id, "response_from_target_agent": response_from_target}
            else:
                # manus_agent_instance.send_a2a_message_to_another_agent should log details of why it returned None
                return {"status": "error", "message": f"Delegation attempt to agent {target_agent_id} did not receive a conclusive response or failed during sending. Check Manus agent logs."}
        except Exception as e: # Catch any unexpected error during the send_a2a_message call itself
            # Consider logging this with manus_agent_instance.logger if available and appropriate
            return {"status": "error", "message": f"An unexpected error occurred while Manus instance was sending the A2A message: {str(e)}"}