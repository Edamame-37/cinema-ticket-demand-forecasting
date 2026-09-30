import os
import subprocess
from typing import Any, Dict, List
import mcp.server.stdio
from mcp.server import Server
from mcp.types import Tool, TextContent

# Initialize the Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    KAGGLE_AVAILABLE = True
except Exception as e:
    KAGGLE_AVAILABLE = False
    print(f"Failed to initialize Kaggle API: {e}")

server = Server("kaggle_mcp")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Kaggle tools."""
    return [
        Tool(
            name="kaggle_search",
            description="Search for Kaggle competitions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword for the competition"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="kaggle_competition_info",
            description="Get basic info and evaluation metric about a competition via CLI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition_name": {"type": "string", "description": "Name of the competition"}
                },
                "required": ["competition_name"]
            }
        ),
        Tool(
            name="kaggle_list_files",
            description="List dataset files for a specific Kaggle competition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition_name": {"type": "string", "description": "Name of the competition"}
                },
                "required": ["competition_name"]
            }
        ),
        Tool(
            name="kaggle_download_data",
            description="Download dataset files for a specific Kaggle competition to a specific path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition_name": {"type": "string", "description": "Name of the competition"},
                    "path": {"type": "string", "description": "Path to download the data to"}
                },
                "required": ["competition_name", "path"]
            }
        ),
        Tool(
            name="kaggle_submit",
            description="Submit predictions to a Kaggle competition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition_name": {"type": "string", "description": "Name of the competition"},
                    "file_path": {"type": "string", "description": "Path to the submission file (e.g., submission.csv)"},
                    "message": {"type": "string", "description": "Submission message"}
                },
                "required": ["competition_name", "file_path", "message"]
            }
        ),
        Tool(
            name="kaggle_leaderboard",
            description="Get the current leaderboard for a Kaggle competition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition_name": {"type": "string", "description": "Name of the competition"}
                },
                "required": ["competition_name"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list:
    """Handle tool calls."""
    if not KAGGLE_AVAILABLE:
        return [TextContent(type="text", text="Error: Kaggle API is not configured or authenticated. Please ensure ~/.kaggle/kaggle.json exists and is valid.")]

    if name == "kaggle_search":
        query = arguments.get("query")
        try:
            competitions = api.competitions_list(search=query)
            result = "\n".join([f"- {c.ref}: {c.title} (Reward: {c.reward})" for c in competitions])
            if not result:
                result = "No competitions found."
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching competitions: {e}")]

    elif name == "kaggle_competition_info":
        competition_name = arguments.get("competition_name")
        try:
            # First get high level info via API search
            comp = next((c for c in api.competitions_list(search=competition_name) if c.ref == competition_name or competition_name in c.ref), None)
            info = f"Competition Info for: {competition_name}\n"
            if comp:
                info += f"Title: {comp.title}\nDescription: {comp.description}\nReward: {comp.reward}\nDeadline: {comp.deadline}\n"
            else:
                info += "Could not fetch detailed title via API search.\n"
            
            # Advice for rules
            info += "\nNOTE: Kaggle API does not fully expose the Markdown rules or evaluation metrics directly. "
            info += "Please use the 'read_browser_page' tool or web search on `https://www.kaggle.com/c/{competition_name}` to read full rules if needed."
            
            return [TextContent(type="text", text=info)]
        except Exception as e:
             return [TextContent(type="text", text=f"Error getting info: {e}")]

    elif name == "kaggle_list_files":
        competition_name = arguments.get("competition_name")
        try:
            files = api.competition_list_files(competition_name)
            result = "\n".join([f"- {f.name} (Size: {f.size})" for f in files])
            if not result:
                result = "No files found."
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing files: {e}")]

    elif name == "kaggle_download_data":
        competition_name = arguments.get("competition_name")
        path = arguments.get("path")
        try:
            os.makedirs(path, exist_ok=True)
            api.competition_download_files(competition_name, path=path, quiet=False)
            return [TextContent(type="text", text=f"Successfully downloaded data for {competition_name} to {path}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error downloading data: {e}")]

    elif name == "kaggle_submit":
        competition_name = arguments.get("competition_name")
        file_path = arguments.get("file_path")
        message = arguments.get("message")
        try:
            api.competition_submit(file_path, message, competition_name)
            return [TextContent(type="text", text=f"Successfully submitted {file_path} to {competition_name}. Check leaderboard for score.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error submitting predictions: {e}")]

    elif name == "kaggle_leaderboard":
        competition_name = arguments.get("competition_name")
        try:
            leaderboard = api.competition_view_leaderboard(competition_name)
            result = "Leaderboard snippet (Top 20):\n"
            for entry in leaderboard[:20]:
                result += f"Rank {entry.get('rank', 'N/A')}: {entry.get('teamName', 'N/A')} - Score: {entry.get('score', 'N/A')}\n"
            return [TextContent(type="text", text=result)]
        except Exception as e:
            try:
                cli_result = subprocess.check_output(["kaggle", "competitions", "leaderboard", competition_name, "--show"], text=True)
                return [TextContent(type="text", text=cli_result)]
            except Exception as cli_e:
                return [TextContent(type="text", text=f"Error fetching leaderboard: {e} | CLI Error: {cli_e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
