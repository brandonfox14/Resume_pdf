import asyncio
from playwright.async_api import async_playwright

URL = "http://localhost:8501"
OUTPUT = "visual_resume.pdf"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1200, "height": 900})
        await page.goto(URL, wait_until="networkidle")

        # Give charts a moment to render
        await page.wait_for_timeout(800)

        await page.pdf(
            path=OUTPUT,
            format="A4",
            print_background=True,
            margin={"top": "8mm", "bottom": "8mm", "left": "8mm", "right": "8mm"},
            scale=0.95,
        )

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
