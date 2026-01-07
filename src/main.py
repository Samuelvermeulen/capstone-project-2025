#!/usr/bin/env python3
"""
Main entry point for Capstone Project.
Run different demonstrations based on command line arguments.
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Capstone Project Demonstrations')
    parser.add_argument('--demo', choices=['basic', 'advanced', 'all'], 
                       default='basic', help='Which demo to run')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA SCIENCE & ADVANCED PROGRAMMING - CAPSTONE PROJECT")
    print("=" * 60)
    
    if args.demo == 'basic' or args.demo == 'all':
        print("\nRunning Basic Concepts Demonstration...")
        print("-" * 40)
        import basics_demo
        print("\n✓ Basic demo completed")
    
    if args.demo == 'advanced' or args.demo == 'all':
        print("\nRunning Advanced Data Analysis...")
        print("-" * 40)
        import data_analysis
        print("\n✓ Advanced analysis completed")
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print(f"Check the '{args.output}' directory for output files.")
    print("=" * 60)

if __name__ == "__main__":
    main()
