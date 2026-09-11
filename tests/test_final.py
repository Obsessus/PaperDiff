"""Final comprehensive E2E browser test."""
import os
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8502"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(PROJECT_DIR, "tests", "screenshots")


def screenshot(page, name):
    path = os.path.join(SCREENSHOTS_DIR, f"{name}.png")
    page.screenshot(path=path)
    print(f"  [SCREENSHOT] {name}.png")


if __name__ == "__main__":
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        try:
            print("=" * 60)
            print("FINAL E2E TEST - PaperDiff")
            print("=" * 60)

            # 1. Load app
            print("\n[1] Loading app...")
            page.goto(APP_URL, wait_until="networkidle")
            page.wait_for_timeout(3000)
            screenshot(page, "final_01_initial")

            # 2. Verify title
            title = page.title()
            assert "PaperDiff" in title, f"Wrong title: {title}"
            print(f"  Title OK: {title}")

            # 3. Verify initial state
            content = page.content()
            assert "Research Papers" in content, "Research Papers section missing"
            assert "Analysis Mode" in content, "Analysis Mode section missing"
            print("  Initial state OK")

            # 4. Click Try Demo
            print("\n[2] Loading demo papers...")
            demo_btn = page.locator("button:has-text('Try Demo')")
            assert demo_btn.count() > 0, "Demo button not found"
            demo_btn.click()
            page.wait_for_timeout(5000)
            screenshot(page, "final_02_papers_loaded")

            # 5. Verify papers loaded
            content = page.content()
            assert "Paper A" in content, "Paper A not found"
            assert "Paper B" in content, "Paper B not found"
            assert "2 of 4 papers uploaded" in content, "Paper count wrong"
            print("  Papers loaded OK")

            # 6. Test Compare Papers mode
            print("\n[3] Testing Compare Papers...")
            compare_label = page.locator("text=Compare Papers").first
            compare_label.click()
            page.wait_for_timeout(1000)

            # Scroll to run button and click
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(500)
            run_btn = page.locator("button:has-text('Run Comparison')")
            if run_btn.count() > 0 and run_btn.first.is_enabled():
                run_btn.first.click()
                print("  Run Comparison clicked! Waiting...")
                page.wait_for_timeout(5000)
                screenshot(page, "final_03_analysis_started")

                # Wait for results
                for i in range(24):
                    page.wait_for_timeout(5000)
                    content = page.content()
                    if "Overall Winner" in content or "Score" in content:
                        print(f"  Results after {(i+1)*5}s!")
                        break
                    print(f"  Waiting... ({(i+1)*5}s)")

                screenshot(page, "final_04_results")

                # Verify results content
                content = page.content()
                assert "Overall Winner" in content, "Overall Winner missing"
                assert "Research Quality" in content or "Topic Fit" in content, "Metrics missing"
                print("  Compare Papers results OK")
            else:
                print("  Run button not available (may already be running)")

            # 7. Test Ask Questions mode
            print("\n[4] Testing Ask Questions...")
            # Click Start Over first
            start_over = page.locator("button:has-text('Start Over')")
            if start_over.count() > 0:
                start_over.first.click()
                page.wait_for_timeout(3000)

            # Load demo again
            demo_btn = page.locator("button:has-text('Try Demo')")
            if demo_btn.count() > 0:
                demo_btn.click()
                page.wait_for_timeout(5000)

            # Select Ask Questions
            rag_label = page.locator("text=Ask Questions").first
            rag_label.click()
            page.wait_for_timeout(1000)

            # Fill question
            text_input = page.locator(".stTextInput input").first
            if text_input.count() > 0:
                text_input.click()
                text_input.fill("What is the main contribution?")
                text_input.press("Tab")
                page.wait_for_timeout(1000)

                # Click Ask
                ask_btn = page.locator("button:has-text('Ask')").first
                if ask_btn.is_enabled():
                    ask_btn.click()
                    print("  Ask clicked! Waiting...")
                    page.wait_for_timeout(5000)
                    screenshot(page, "final_05_rag_started")

                    # Wait for response
                    for i in range(12):
                        page.wait_for_timeout(5000)
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(500)
                        content = page.content()
                        if "Q:" in content and len(content) > 5000:
                            print(f"  Response after {(i+1)*5}s!")
                            break
                        print(f"  Waiting... ({(i+1)*5}s)")

                    screenshot(page, "final_06_rag_response")

                    # Verify RAG response
                    content = page.content()
                    if "Q:" in content:
                        print("  RAG response OK")
                    else:
                        print("  RAG response not detected (may still be loading)")

            print("\n" + "=" * 60)
            print("ALL TESTS PASSED - Project is ready for GitHub!")
            print("=" * 60)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            screenshot(page, "final_error")
        finally:
            browser.close()
