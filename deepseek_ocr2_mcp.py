import os
import asyncio
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

# -----------------------------
# Configuration
# -----------------------------
OCR_SERVER_URL = os.getenv("DS_OCR2_URL", "http://127.0.0.1:8012")
MCP_SERVER_NAME = "deepseek-ocr2-mcp"

# Create FastMCP server instance
mcp = FastMCP(MCP_SERVER_NAME)


@mcp.tool()
async def ocr_image(
    image_path: str, mode: str = "markdown", keep_files: bool = False
) -> str:
    """
    Perform OCR on an image using the DeepSeek-OCR-2 server.

    Args:
        image_path: Absolute path to the image file.
        mode: OCR mode, either "markdown" (default) or "free".
        keep_files: Whether to keep intermediate files on the server (default False).

    Returns:
        The extracted text from the image.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    timeout = httpx.Timeout(
        600.0, connect=10.0
    )  # Long timeout for model loading/inference

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # check health first
            try:
                resp = await client.get(f"{OCR_SERVER_URL}/health")
                resp.raise_for_status()
            except Exception as e:
                return f"Error: Could not connect to OCR server at {OCR_SERVER_URL}. Is it running? Details: {e}"

            # Prepare upload
            files = {"file": open(image_path, "rb")}
            data = {"mode": mode, "keep_files": str(keep_files).lower()}

            resp = await client.post(f"{OCR_SERVER_URL}/v1/ocr", files=files, data=data)

            if resp.status_code != 200:
                return f"Error from OCR server (Status {resp.status_code}): {resp.text}"

            result = resp.json()
            return result.get("text", "")

    except Exception as e:
        return f"Error performing OCR: {str(e)}"


if __name__ == "__main__":
    mcp.run()
