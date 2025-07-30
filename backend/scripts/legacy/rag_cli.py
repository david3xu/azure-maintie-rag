#!/usr/bin/env python3
"""
RAG CLI - Unified Command Line Interface
Single entry point for all RAG operations
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all tool modules
from scripts import azure_setup
from scripts import data_pipeline
from scripts import gnn_trainer
from scripts import test_runner
from scripts import workflow_runner
from scripts import demo_runner


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='rag-cli',
        description='Azure Universal RAG CLI - Unified tool for all operations'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Azure setup command
    setup_parser = subparsers.add_parser('setup', help='Setup and validate Azure configuration')
    
    # Data pipeline command
    data_parser = subparsers.add_parser('data', help='Run data processing pipeline')
    data_parser.add_argument('--mode', choices=['full', 'upload', 'extract', 'train', 'query'], 
                           default='full', help='Pipeline mode (default: full)')
    data_parser.add_argument('--domain', default='general', help='Domain name')
    data_parser.add_argument('--source', default='data/raw', help='Source data path')
    
    # GNN training command
    gnn_parser = subparsers.add_parser('train', help='Train GNN model')
    gnn_parser.add_argument('--model', default='latest', help='Model version')
    
    # Test validation command
    test_parser = subparsers.add_parser('test', help='Run tests and validation')
    test_parser.add_argument('--type', choices=['unit', 'integration', 'all'], default='all')
    
    # Workflow analysis command
    workflow_parser = subparsers.add_parser('workflow', help='Analyze and run workflows')
    workflow_parser.add_argument('action', choices=['lifecycle', 'query', 'cleanup'])
    workflow_parser.add_argument('--query', help='Query text for query action')
    
    # Demo runner command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--scenario', default='default', help='Demo scenario')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate tool
    if args.command == 'setup':
        return asyncio.run(azure_setup.main())
    
    elif args.command == 'data':
        # Pass arguments to data pipeline
        original_argv = sys.argv
        sys.argv = ['data_pipeline.py', '--mode', args.mode, '--domain', args.domain, '--source', args.source]
        result = asyncio.run(data_pipeline.main())
        sys.argv = original_argv
        return result
    
    elif args.command == 'train':
        return asyncio.run(gnn_trainer.main())
    
    elif args.command == 'test':
        return asyncio.run(test_runner.main())
    
    elif args.command == 'workflow':
        # Pass action to workflow runner
        original_argv = sys.argv
        if args.action == 'query' and args.query:
            sys.argv = ['workflow_runner.py', args.action, args.query]
        else:
            sys.argv = ['workflow_runner.py', args.action]
        result = asyncio.run(workflow_runner.main())
        sys.argv = original_argv
        return result
    
    elif args.command == 'demo':
        return asyncio.run(demo_runner.main())
    
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())