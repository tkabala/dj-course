"""
MCP server exposing Azor conversation management tools.

Register in Claude Code / Claude Desktop:
{
  "mcpServers": {
    "azor": {
      "command": "python",
      "args": ["/home/tomek/dj-course/M1/azor-chatdog-py/src/mcp_server.py"]
    }
  }
}
"""

import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from files import session_files

mcp = FastMCP("azor")


@mcp.tool()
def list_conversations() -> list[dict]:
    """List all Azor conversations with id, title, message count, and last activity."""
    return session_files.list_sessions()


@mcp.tool()
def get_conversation(session_id: str) -> dict:
    """
    Get full conversation history and metadata for a given session_id.

    Args:
        session_id: The session identifier (e.g. '550e8400-e29b-41d4-a716-446655440000')
    """
    history, title, error = session_files.load_session_history(session_id)
    if error:
        return {"error": error}

    # Also read raw JSON for model/system_role metadata
    from files.config import LOG_DIR
    import json
    log_path = os.path.join(LOG_DIR, f"{session_id}-log.json")
    metadata = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            metadata = {
                "model": raw.get("model"),
                "system_role": raw.get("system_role"),
            }
        except Exception:
            pass

    return {
        "session_id": session_id,
        "title": title,
        "history": history,
        **metadata,
    }


@mcp.tool()
def delete_conversation(session_id: str) -> dict:
    """
    Delete an Azor conversation by session_id.

    Args:
        session_id: The session identifier to delete
    """
    success, error = session_files.remove_session_file(session_id)
    if success:
        return {"success": True, "message": f"Session '{session_id}' deleted."}
    return {"success": False, "error": error}


if __name__ == "__main__":
    mcp.run()
