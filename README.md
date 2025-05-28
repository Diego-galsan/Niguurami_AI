# Nigúurami
Nigúurami is a Tarahumara word that translates to "helper". True to its name, this AI agent is designed to support and assist in multi-agent interactions.
Developed on top of the OpenManus project, it features a simplified version of the A2A (Agent-to-Agent) communication protocol.


- You can test it using Postman
- Get :  
   - http://localhost:8000/.well-known/agent-card.json. 
   - http://localhost:8000/discovery/agents  
   - http://localhost:8000/discovery/agents/{{Agentid}}

- Post:  
   - http://localhost:8000/{{Agentid}}/message?Content-Type
       
   - Header:  
         Key: Content-Type  
         Values: application/json  
   - Body:  
        raw:  
                  {  
          "sender_agent_id": "postman-test-suite-001",  
          "message_type": "execute_capability",  
          "payload": {  
            "capability_name": "execute_prompt",  
            "params": {  
              "prompt": "Base on your knowladge base, what year México got the independency, just give me the indepence date and that's all."  
            }  
          }  
        }  
## How to test the A2A implementation?

Clone the main brach for one of your Agents, this can be Agent 1. Clone the Agent2 branch for your second agent Agent 2.
Run both servers. For this example ports are 8000 and 8001

- uvicorn fastapi_a2a_server:app --reload
- uvicorn fastapi_a2a_server:app --reload --port 8001

Find the information of Agent 2 via the endpoint. Copy and register the Agent 2 in Agent 1.
<img width="1297" alt="imagen" src="https://github.com/user-attachments/assets/02b1a8f4-10b7-4690-b730-2f73a4c40a74" />

Register Agent 2 in Agent 1. This is necessary to allow Agent 1 to know about Agent 2.
<img width="1299" alt="imagen" src="https://github.com/user-attachments/assets/38a96316-3dc9-4631-8e04-816af4e722e4" />

Run a prompt that needs the participation of Agent 2
<img width="1295" alt="imagen" src="https://github.com/user-attachments/assets/b8099f90-bd64-491d-a377-d456498b459a" />

And That's all :) be happy!!.

