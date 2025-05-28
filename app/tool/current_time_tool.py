# File: app/tool/time_tool.py
import datetime
from typing import TYPE_CHECKING, Optional
from app.tool import BaseTool
# For Python 3.9+
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # zoneinfo is not available (Python < 3.9).
    # For production with older Python, you would typically use the 'pytz' library.
    # For this example, we'll set them to None and handle it in the execute method.
    ZoneInfo = None
    ZoneInfoNotFoundError = None # type: ignore

from pydantic import BaseModel, Field
from app.logger import logger # Assuming your logger is here

if TYPE_CHECKING:
    from app.agent.manus import Manus # For type hinting 'agent' if your framework passes it

# --- Input Schema for the Tool ---
class GetCurrentTimeInput(BaseModel):
    timezone_identifier: Optional[str] = Field(
        default=None,
        description="Optional IANA timezone name (e.g., 'America/New_York', 'Europe/Paris', 'Asia/Tokyo', 'UTC', 'America/Mexico_City'). If not provided, returns current UTC and the server's local time."
    )

# --- Tool Definition ---
# IMPORTANT: This tool should inherit from the same base class as your other working tools
# to ensure it has any required methods like `to_param()`.
# Replace `YourBaseToolClass` with the actual base class from your framework.
# If your tools just inherit from pydantic.BaseModel and that worked for others,
# then `GetCurrentTimeTool(BaseModel)` is fine.

# from app.tool.base import YourBaseToolClass # Example: uncomment and use your actual base
class GetCurrentTimeTool(BaseTool): # <<< !!! Ensure this inherits from your agent's actual base tool class !!!
    name: str = "get_current_time"
    description: str = (
        "Fetches the current time. It can provide the time in a specific IANA timezone "
        "if specified (e.g., 'America/New_York', 'Europe/London', 'UTC'). "
        "If no timezone is specified, it returns the current Coordinated Universal Time (UTC) "
        "and the server's local time."
    )
    args_schema: type[BaseModel] = GetCurrentTimeInput

    async def execute(self, agent: 'Manus', args: GetCurrentTimeInput) -> str:
        # The 'agent' parameter might not be strictly needed for this specific tool's core logic
        # but is kept for consistency with other tools if your framework passes it.

        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if args.timezone_identifier:
            if not ZoneInfo:
                logger.warning("GetCurrentTimeTool: zoneinfo module not available (requires Python 3.9+). Cannot get time for a specific timezone. Consider installing 'pytz' for older Python versions.")
                return ("Error: Timezone-specific time retrieval is not supported by the server's current Python environment "
                        "(Python 3.9+ and its 'zoneinfo' module are typically needed). You can ask for UTC or server local time instead.")
            try:
                # Validate common request for "local" if user is in Mexico City based on context
                effective_tz_id = args.timezone_identifier
                if args.timezone_identifier.lower() in ["local", "current", "my timezone"] and agent.name == "Manus": # A bit of a heuristic
                    # Based on your provided context: "Ciudad López Mateos, State of Mexico, Mexico"
                    # The LLM might infer this based on your location context and ask for local time.
                    logger.info(f"GetCurrentTimeTool: Interpreting '{args.timezone_identifier}' as 'America/Mexico_City' based on contextual location.")
                    effective_tz_id = "America/Mexico_City" # Example for your context

                target_tz = ZoneInfo(effective_tz_id)
                now_in_timezone = now_utc.astimezone(target_tz)
                return f"The current time in {effective_tz_id} is {now_in_timezone.strftime('%A, %B %d, %Y %I:%M:%S %p %Z (%z)')}."
            except ZoneInfoNotFoundError: # type: ignore
                logger.warning(f"GetCurrentTimeTool: Timezone identifier '{args.timezone_identifier}' (effective: '{effective_tz_id}') not found.")
                return (f"Error: The timezone identifier '{args.timezone_identifier}' was not recognized. "
                        "Please use a valid IANA timezone name (e.g., 'America/New_York', 'UTC', 'Europe/Paris').")
            except Exception as e:
                logger.error(f"GetCurrentTimeTool: Error processing timezone '{args.timezone_identifier}' (effective: '{effective_tz_id}'): {e}", exc_info=True)
                return f"Error: Could not determine the time for the timezone '{args.timezone_identifier}'."
        else:
            # Default: Provide UTC and server's local time
            now_local = datetime.datetime.now().astimezone() # Get current time in server's local timezone
            return (f"Current Coordinated Universal Time (UTC) is {now_utc.strftime('%A, %B %d, %Y %I:%M:%S %p %Z')}. "
                    f"The server's current local time is {now_local.strftime('%A, %B %d, %Y %I:%M:%S %p %Z (%z)')}.")