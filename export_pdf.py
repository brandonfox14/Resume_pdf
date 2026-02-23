import asyncio
import argparse
from pathlib import Path
from playwright.async_api import async_playwright

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8501", help="Streamlit URL")
    p.add_argument("--out", default="visual_resume.pdf", help="Output PDF path")
    return p.parse_args()

async def main():
    args = parse_args()
    out_path = Path(args.out)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1200, "height": 900})
        await page.goto(args.url, wait_until="networkidle")

        # Let matplotlib canvases settle
        await page.wait_for_timeout(900)

        # Print to PDF
        await page.pdf(
            path=str(out_path),
            format="Letter",          # change to "A4" if you prefer
            print_background=True,
            margin={"top": "8mm", "bottom": "8mm", "left": "8mm", "right": "8mm"},
            scale=0.92                # tweak this if needed to force 1-page
        )

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

    