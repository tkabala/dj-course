"""
MCP client manager for Azor.

Loads server configurations from ~/.azor/mcp_servers.json and provides
sync wrappers around the async MCP client API.

Config format (~/.azor/mcp_servers.json):
[
  {
    "name": "azor-conversations",
    "command": "python",
    "args": ["/path/to/src/mcp_server.py"]
  }
]
"""

import asyncio
import json
import os
from typing import Any

from llm.tools import ToolDefinition

MCP_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".azor", "mcp_servers.json")

_DEFAULT_CONFIG: list[dict] = []


def _ensure_config_exists() -> None:
    """Create the config file with an empty list if it doesn't exist."""
    if not os.path.exists(MCP_CONFIG_PATH):
        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CONFIG, f, indent=2)


def _load_config() -> list[dict]:
    """Load and return server configs from the config file."""
    _ensure_config_exists()
    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


class MCPClientManager:
    """
    Manages connections to configured MCP servers.

    Fetches tool lists on demand and provides a sync interface for
    calling MCP tools from Azor's synchronous code.
    """

    def __init__(self) -> None:
        self._server_configs: list[dict] = _load_config()
        # Cached: server_name -> list[ToolDefinition]
        self._tools_cache: dict[str, list[ToolDefinition]] = {}

    def has_servers(self) -> bool:
        """Return True if at least one server is configured."""
        return bool(self._server_configs)

    def get_all_tools(self) -> list[ToolDefinition]:
        """
        Return all tools from all configured servers, namespaced as
        mcp__{server_name}__{tool_name}.
        """
        all_tools: list[ToolDefinition] = []
        for config in self._server_configs:
            server_name = config.get("name", "unknown")
            tools = self._get_server_tools(config)
            self._tools_cache[server_name] = tools
            all_tools.extend(tools)
        return all_tools

    def _get_server_tools(self, config: dict) -> list[ToolDefinition]:
        """Connect to a single server and return its tools as ToolDefinitions."""
        server_name = config.get("name", "unknown")
        command = config.get("command", "python")
        args = config.get("args", [])

        try:
            return asyncio.run(self._async_get_tools(server_name, command, args))
        except Exception as e:
            print(f"[MCP] Warning: could not connect to server '{server_name}': {e}")
            return []

    async def _async_get_tools(
        self, server_name: str, command: str, args: list[str]
    ) -> list[ToolDefinition]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(command=command, args=args)
        async with stdio_client(server_params, errlog=open(os.devnull, 'w')) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                result: list[ToolDefinition] = []
                for tool in tools_result.tools:
                    namespaced_name = f"mcp__{server_name}__{tool.name}"
                    # MCP input schema is already JSON Schema compatible
                    parameters = tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                    }
                    result.append(
                        ToolDefinition(
                            name=namespaced_name,
                            description=tool.description or "",
                            parameters=parameters,
                        )
                    )
                return result

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a tool on a named MCP server synchronously.

        Args:
            server_name: The server name (as in config "name" field)
            tool_name: The bare tool name (without mcp__ prefix)
            arguments: Tool arguments

        Returns:
            Result dict with at least a 'content' key
        """
        config = next(
            (c for c in self._server_configs if c.get("name") == server_name), None
        )
        if config is None:
            return {"error": f"MCP server '{server_name}' not found in config"}

        command = config.get("command", "python")
        args = config.get("args", [])

        try:
            return asyncio.run(
                self._async_call_tool(command, args, tool_name, arguments)
            )
        except Exception as e:
            return {"error": f"MCP tool call failed: {e}"}

    async def _async_call_tool(
        self,
        command: str,
        args: list[str],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(command=command, args=args)
        async with stdio_client(server_params, errlog=open(os.devnull, 'w')) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                # Extract text content from result
                content_parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content_parts.append(item.text)
                return {"content": "\n".join(content_parts)}
