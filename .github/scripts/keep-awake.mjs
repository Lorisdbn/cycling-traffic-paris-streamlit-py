// Opens the Streamlit app in a real headless browser so that Streamlit
// Community Cloud records a genuine visit (WebSocket session), wakes the app
// up if it is asleep, and waits until the app has actually rendered.
import { chromium } from "playwright";

const APP_URL = process.env.APP_URL ?? "https://cycling-traffic-paris.streamlit.app/";
const WAKE_BUTTON = /get this app back up/i;
const APP_SELECTOR = '[data-testid="stApp"], [data-testid="stAppViewContainer"]';
const READY_TIMEOUT_MS = 4 * 60_000; // a cold start can take a few minutes
const STAY_CONNECTED_MS = 30_000;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The app may be rendered in the page itself or inside an iframe.
async function appIsRendered(page) {
  for (const frame of page.frames()) {
    try {
      if (await frame.locator(APP_SELECTOR).first().isVisible()) return true;
    } catch {
      // frame detached while checking: ignore and keep polling
    }
  }
  return false;
}

const browser = await chromium.launch();
const page = await browser.newPage();
let exitCode = 0;

try {
  console.log(`Opening ${APP_URL}`);
  await page.goto(APP_URL, { waitUntil: "domcontentloaded", timeout: 90_000 });

  const deadline = Date.now() + READY_TIMEOUT_MS;
  let wokeUp = false;
  while (!(await appIsRendered(page))) {
    if (Date.now() > deadline) throw new Error("App did not render in time");

    const wakeButton = page.getByRole("button", { name: WAKE_BUTTON });
    if (!wokeUp && (await wakeButton.isVisible().catch(() => false))) {
      console.log("App is asleep: clicking the wake-up button");
      await wakeButton.click();
      wokeUp = true;
    }
    await sleep(5_000);
  }

  console.log(wokeUp ? "App woken up and rendered" : "App is awake and rendered");
  await sleep(STAY_CONNECTED_MS); // keep the session open so the visit counts
} catch (err) {
  console.error(`Keep-awake failed: ${err.message}`);
  exitCode = 1;
} finally {
  await page.screenshot({ path: "keep-awake.png", fullPage: true }).catch(() => {});
  await browser.close();
}

process.exit(exitCode);
