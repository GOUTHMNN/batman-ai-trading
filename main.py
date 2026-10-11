"""
🦇 BATMAN AI TRADING SYSTEM — CLI ENTRY POINT
================================================
Command-line interface to run any module.
Usage:
    python main.py --train --ticker AAPL
    python main.py --predict --ticker TSLA
    python main.py --backtest
    python main.py --deploy
    python main.py --train --predict --backtest --ticker AAPL
"""

import os
import sys
import argparse
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import load_config, run_pipeline
from model_core import run_training
from signals_risk import run_signal_pipeline
from backtest_engine import run_full_backtest
from deploy_night_patrol import daily_patrol, start_scheduler

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanMain")


def print_banner():
    """Print the Batman trading system banner."""
    banner = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          🦇  B A T M A N   A I   T R A D I N G  🦇       ║
    ║                                                          ║
    ║       "It's not who I am underneath,                     ║
    ║        but what I do that defines me."                    ║
    ║                                                          ║
    ║       LSTM · Signals · Backtest · Night Patrol           ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main CLI entry point with argument parsing."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="🦇 Batman AI Trading System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline --ticker AAPL     Run data pipeline only
  python main.py --train --ticker AAPL        Train LSTM model
  python main.py --predict --ticker AAPL      Generate prediction & signal
  python main.py --backtest                   Run backtest on test set
  python main.py --deploy                     Start Night Patrol scheduler
  python main.py --deploy-now                 Run one-shot patrol immediately
  python main.py --train --predict --backtest Full pipeline
        """,
    )

    parser.add_argument("--ticker", type=str, default=None,
                        help="Ticker symbol (default: config.yaml value)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run data pipeline (fetch + features + sequences)")
    parser.add_argument("--train", action="store_true",
                        help="Train LSTM model")
    parser.add_argument("--predict", action="store_true",
                        help="Generate prediction and trading signal")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest simulation")
    parser.add_argument("--deploy", action="store_true",
                        help="Start Night Patrol scheduler (daily at 03:00 GMT)")
    parser.add_argument("--deploy-now", action="store_true",
                        help="Run one-shot Night Patrol immediately")

    args = parser.parse_args()

    # If no args, show help
    if not any([args.pipeline, args.train, args.predict, args.backtest,
                args.deploy, args.deploy_now]):
        parser.print_help()
        return

    config = load_config()

    # Override ticker if provided
    if args.ticker:
        config["ticker"] = args.ticker
        logger.info(f"🎯 Ticker override: {args.ticker}")

    try:
        # ─── Data Pipeline ───
        if args.pipeline:
            logger.info("=" * 60)
            logger.info("🦇 PHASE: DATA PIPELINE")
            logger.info("=" * 60)
            run_pipeline(ticker=args.ticker, config=config)

        # ─── Training ───
        if args.train:
            logger.info("=" * 60)
            logger.info("🦇 PHASE: MODEL TRAINING")
            logger.info("=" * 60)
            result = run_training(ticker=args.ticker, config=config)
            eval_r = result["eval_results"]
            print(f"\n   📊 Test MAE: ${eval_r['mae']:.2f}")
            print(f"   📊 Test RMSE: ${eval_r['rmse']:.2f}")

        # ─── Prediction ───
        if args.predict:
            logger.info("=" * 60)
            logger.info("🦇 PHASE: PREDICTION & SIGNALS")
            logger.info("=" * 60)
            result = run_signal_pipeline(ticker=args.ticker, config=config)
            print(f"\n   🎯 Signal: {result['signal']}")
            print(f"   📈 Predicted: {result['predicted_close']:.2f}")
            print(f"   📊 Confidence: {result['confidence']*100:.1f}%")

        # ─── Backtest ───
        if args.backtest:
            logger.info("=" * 60)
            logger.info("🦇 PHASE: BACKTEST")
            logger.info("=" * 60)
            _, metrics, _ = run_full_backtest(config=config)
            print(f"\n   📊 Backtest Results:")
            for k, v in metrics.items():
                print(f"      {k}: {v}")

        # ─── Deploy ───
        if args.deploy:
            logger.info("=" * 60)
            logger.info("🦇 PHASE: NIGHT PATROL DEPLOYMENT")
            logger.info("=" * 60)
            start_scheduler(config=config)

        if args.deploy_now:
            logger.info("=" * 60)
            logger.info("🦇 PHASE: IMMEDIATE PATROL")
            logger.info("=" * 60)
            daily_patrol(config=config)

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("💡 Run --pipeline and --train first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
