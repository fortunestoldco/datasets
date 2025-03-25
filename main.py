#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import traceback
import subprocess
import importlib.util

# Setup logging first
def setup_logging(debug: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        debug (bool): Enable debug logging if True
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("doc_downloader.log"),
            logging.StreamHandler()
        ]
    )

# Command line argument parsing
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='SDK Documentation Dataset Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--no-tui', action='store_true', 
                       help='Run in command line mode without TUI')
    parser.add_argument('--org', type=str, 
                       help='Organization to process')
    parser.add_argument('--output', type=str, 
                       default='./downloaded_docs',
                       help='Output directory')
    parser.add_argument('--test-ratio', type=float, 
                       default=0.2,
                       help='Test split ratio')
    parser.add_argument('--github-token', type=str,
                       default=os.getenv('GITHUB_TOKEN'),
                       help='GitHub API token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--hf-token', type=str,
                       default=os.getenv('HF_TOKEN'),
                       help='Hugging Face API token (or set HF_TOKEN env var)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--retry-limit', type=int,
                       default=3,
                       help='Number of retries for API calls')
    parser.add_argument('--retry-delay', type=int,
                       default=60,
                       help='Delay in seconds between retries')
    return parser.parse_args()

# Check and install required libraries
def check_dependencies():
    """Check and install required dependencies."""
    logger = logging.getLogger(__name__)
    
    REQUIRED_LIBRARIES = [
        "huggingface_hub", "datasets", "nbformat", "nbconvert",
        "gitpython", "tqdm", "textual", "python-dotenv"
    ]
    
    missing_libs = []
    for lib in REQUIRED_LIBRARIES:
        # Check if lib is installed
        lib_name = lib.split('[')[0] if '[' in lib else lib
        spec = importlib.util.find_spec(lib_name)
        if spec is None:
            missing_libs.append(lib)
    
    if missing_libs:
        logger.info(f"Installing missing libraries: {', '.join(missing_libs)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_libs)
            logger.info("All required libraries installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error("Please install the following libraries manually:")
            for lib in missing_libs:
                logger.error(f"  pip install {lib}")
            sys.exit(1)

def main_tui() -> int:
    """
    Main function with TUI interface.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.debug)
        logger = logging.getLogger(__name__)
        
        # Check and install dependencies
        check_dependencies()
        
        # Import after dependencies are checked
        import dotenv
        from config import load_or_create_config, save_config
        from processors import process_all_repositories
        from ui.app import MainTUIApp
        
        # Load environment variables
        dotenv.load_dotenv()
        
        # Load or create configuration
        config = load_or_create_config()
        
        # Update config with command line arguments if provided
        if args.github_token:
            config['github_token'] = args.github_token
        if args.hf_token:
            config['huggingface_token'] = args.hf_token
        if args.output:
            config['output_directory'] = args.output
        if args.test_ratio is not None:
            config['test_ratio'] = args.test_ratio
            
        # If organization is provided via command line, add it to repositories
        if args.org:
            # Check if organization already exists in repositories
            existing_orgs = [
                repo['name'] for repo in config['repositories'] 
                if repo.get('type') == 'organization' and repo.get('name') == args.org
            ]
            
            if not existing_orgs:
                config['repositories'].append({
                    'type': 'organization',
                    'name': args.org,
                    'source': 'github'
                })
                logger.info(f"Added organization {args.org} to repositories")
            
        # Save updated config
        save_config(config)
        
        # If no-tui flag is set and org is provided, run in command line mode
        if args.no_tui and args.org:
            logger.info(f"Running in command line mode for organization: {args.org}")
            result = process_all_repositories(config)
            return 0 if result else 1
            
        # Otherwise, start TUI
        app = MainTUIApp(config)
        app.run()
        return 0
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Program interrupted by user")
        return 130
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

def main() -> int:
    """
    Legacy main function (kept for backward compatibility).
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        print("Information: This script now uses a text-based user interface (TUI). Switching to TUI mode...")
        return main_tui()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
