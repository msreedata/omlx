# SPDX-License-Identifier: Apache-2.0
"""
MCP (Model Context Protocol) API routes.

This module provides FastAPI routes for MCP tool management:
- GET /v1/mcp/tools - List available MCP tools
- GET /v1/mcp/servers - List MCP server status
- POST /v1/mcp/execute - Execute an MCP tool

All endpoints require API key authentication (when configured).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .openai_models import (
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPToolInfo,
    MCPToolsResponse,
)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])

_security = HTTPBearer(auto_error=False)

# Callback function to get MCP manager (set by server.py)
_get_mcp_manager = None


def set_mcp_manager_getter(getter):
    """
    Set the callback function to get MCP manager.

    Args:
        getter: A callable that returns the MCP manager instance or None
    """
    global _get_mcp_manager
    _get_mcp_manager = getter


def _get_manager():
    """Get the MCP manager instance."""
    if _get_mcp_manager is None:
        return None
    return _get_mcp_manager()


async def _require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> bool:
    """Verify API key for MCP endpoints.

    Lazily imports verify_api_key from server module to avoid circular
    imports (server.py imports this module at startup).
    """
    from omlx.server import verify_api_key

    return await verify_api_key(request, credentials)


@router.get("/tools")
async def list_mcp_tools(_: bool = Depends(_require_api_key)) -> MCPToolsResponse:
    """List all available MCP tools."""
    manager = _get_manager()
    if manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in manager.get_all_tools():
        tools.append(MCPToolInfo(
            name=tool.full_name,
            description=tool.description,
            server=tool.server_name,
            parameters=tool.input_schema,
        ))

    return MCPToolsResponse(tools=tools, count=len(tools))


@router.get("/servers")
async def list_mcp_servers(_: bool = Depends(_require_api_key)) -> MCPServersResponse:
    """Get status of all MCP servers."""
    manager = _get_manager()
    if manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in manager.get_server_status():
        servers.append(MCPServerInfo(
            name=status.name,
            state=status.state.value,
            transport=status.transport.value,
            tools_count=status.tools_count,
            error=status.error,
        ))

    return MCPServersResponse(servers=servers)


@router.post("/execute")
async def execute_mcp_tool(
    request: MCPExecuteRequest,
    _: bool = Depends(_require_api_key),
) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    manager = _get_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="MCP not configured. Start server with --mcp-config"
        )

    result = await manager.execute_tool(
        request.tool_name,
        request.arguments,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )
