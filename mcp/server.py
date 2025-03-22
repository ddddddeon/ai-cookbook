import asyncio
from shodan import Shodan
from shodan.cli.helpers import get_api_key
from typing import Any
import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-server")


@mcp.tool()
async def search_shodan(term: str) -> str:
    shodan = Shodan(os.getenv("SHODAN_API_KEY"))
    limit = 100
    counter = 0

    results = []

    try:
        response = shodan.search(term)
    except Exception as e:
        return str(e)

    for result in response["matches"]:
        if counter >= limit:
            break

        result_str = "\n".join([result.get("product", "")])
        results.append(result_str)
        counter += 1

    return "\n\n".join([str(result) for result in results])


if __name__ == "__main__":
    mcp.run(transport="stdio")
