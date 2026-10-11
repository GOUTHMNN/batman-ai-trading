"""
🦇 BATMAN AI TRADING SYSTEM — NIGHT PATROL DEPLOYMENT
========================================================
Scheduled daily execution: fetch → predict → signal → email.
Gmail SMTP with TLS. Runs at 03:00 GMT. The Dark Knight never sleeps.
"""

import os
import sys
import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import schedule
import yaml

from data_pipeline import load_config
from signals_risk import run_signal_pipeline

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanNightPatrol")


def send_bat_signal_email(signal_result, config=None):
    """
    Send Bat-Signal email alert via Gmail SMTP.

    Args:
        signal_result: Dict from run_signal_pipeline
        config: Config dict with SMTP settings

    Returns:
        bool: True if sent successfully
    """
    if config is None:
        config = load_config()

    smtp_server = config.get("smtp_server", "smtp.gmail.com")
    smtp_port = config.get("smtp_port", 587)
    email_from = config.get("email_from", "")
    email_to = config.get("email_to", "")
    email_password = config.get("email_password", "")

    if not email_from or not email_to or not email_password or email_password == "YOUR_APP_PASSWORD_HERE":
        logger.warning("⚠️ Email not configured. Set BATMAN_EMAIL_FROM, BATMAN_EMAIL_TO and BATMAN_EMAIL_PASSWORD.")
        return False

    ticker = signal_result.get("ticker", "UNKNOWN")
    signal = signal_result.get("signal", "UNKNOWN")
    predicted = signal_result.get("predicted_close", 0)
    current = signal_result.get("current_close", 0)
    confidence = signal_result.get("confidence", 0)
    pct_change = signal_result.get("pct_change", 0)
    stop_loss = signal_result.get("stop_loss", 0)
    take_profit = signal_result.get("take_profit", 0)
    shares = signal_result.get("shares", 0)

    # Signal emoji
    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")

    subject = f"🦇 Bat-Signal: {signal} {ticker} @ {predicted:.2f}"

    html_body = f"""
    <html>
    <body style="background:#0a0a0a; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:20px;">
        <div style="max-width:600px; margin:auto; background:#1a1a2e; border-radius:12px; padding:30px;
                    border:1px solid #333; box-shadow:0 0 20px rgba(0,170,255,0.2);">
            <h1 style="color:#f1c40f; text-align:center; font-size:28px; margin-bottom:5px;">
                🦇 BAT-SIGNAL ALERT
            </h1>
            <p style="text-align:center; color:#888; margin-bottom:25px;">
                {datetime.now().strftime('%Y-%m-%d %H:%M GMT')}
            </p>

            <div style="text-align:center; background:#0a0a0a; border-radius:8px; padding:20px; margin:15px 0;
                        border:2px solid {'#2ecc71' if signal == 'BUY' else '#c0392b' if signal == 'SELL' else '#f1c40f'};">
                <h2 style="color:{'#2ecc71' if signal == 'BUY' else '#c0392b' if signal == 'SELL' else '#f1c40f'};
                           font-size:36px; margin:0;">
                    {signal_emoji} {signal}
                </h2>
                <p style="color:#888; font-size:16px; margin:5px 0 0;">{ticker}</p>
            </div>

            <table style="width:100%; border-collapse:collapse; margin:20px 0;">
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px; color:#888;">Current Price</td>
                    <td style="padding:10px; text-align:right; color:#00aaff; font-weight:bold;">
                        {current:.2f}</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px; color:#888;">Predicted Close</td>
                    <td style="padding:10px; text-align:right; color:#f1c40f; font-weight:bold;">
                        {predicted:.2f}</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px; color:#888;">Expected Change</td>
                    <td style="padding:10px; text-align:right;
                               color:{'#2ecc71' if pct_change > 0 else '#c0392b'}; font-weight:bold;">
                        {pct_change:+.2f}%</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px; color:#888;">Confidence</td>
                    <td style="padding:10px; text-align:right; color:#e0e0e0; font-weight:bold;">
                        {confidence*100:.1f}%</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px; color:#888;">Stop-Loss</td>
                    <td style="padding:10px; text-align:right; color:#c0392b;">{stop_loss:.2f}</td>
                </tr>
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px; color:#888;">Take-Profit</td>
                    <td style="padding:10px; text-align:right; color:#2ecc71;">{take_profit:.2f}</td>
                </tr>
                <tr>
                    <td style="padding:10px; color:#888;">Position Size</td>
                    <td style="padding:10px; text-align:right; color:#e0e0e0;">{shares} shares</td>
                </tr>
            </table>

            <p style="text-align:center; color:#555; font-size:12px; margin-top:25px;">
                Batman AI Trading System • Night Patrol • Automated Signal
            </p>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)
        logger.info(f"📧 Bat-Signal email sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"❌ Email failed: {e}")
        return False


def daily_patrol(config=None):
    """
    Execute the daily patrol: fetch latest → predict → signal → email → log.

    Args:
        config: Config dict
    """
    if config is None:
        config = load_config()

    logger.info(f"{'='*60}")
    logger.info("🌙 NIGHT PATROL — STARTING DAILY SWEEP")
    logger.info(f"{'='*60}")

    try:
        # Run signal pipeline
        signal_result = run_signal_pipeline(config=config)

        # Send email
        send_bat_signal_email(signal_result, config)

        logger.info("🌙 NIGHT PATROL — SWEEP COMPLETE")

    except Exception as e:
        logger.error(f"❌ Night Patrol failed: {e}")

        # Try to send error notification
        try:
            error_result = {
                "ticker": config.get("ticker", "UNKNOWN"),
                "signal": "ERROR",
                "predicted_close": 0,
                "current_close": 0,
                "confidence": 0,
                "pct_change": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "shares": 0,
            }
            send_bat_signal_email(error_result, config)
        except Exception:
            pass


def start_scheduler(config=None):
    """
    Start the Night Patrol scheduler. Runs daily at configured time (GMT).

    Args:
        config: Config dict
    """
    if config is None:
        config = load_config()

    run_time = config.get("schedule_time", "03:00")

    logger.info(f"{'='*60}")
    logger.info(f"🌙 NIGHT PATROL SCHEDULER — Armed for {run_time} GMT daily")
    logger.info(f"{'='*60}")

    schedule.every().day.at(run_time).do(daily_patrol, config=config)

    logger.info("⏰ Scheduler running. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("🛑 Night Patrol stood down.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        daily_patrol()
    else:
        start_scheduler()
