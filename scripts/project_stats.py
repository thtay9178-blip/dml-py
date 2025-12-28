#!/usr/bin/env python
"""
Project Statistics Script

Generates statistics about the DML-PY codebase.
"""

import os
from pathlib import Path


def count_lines(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0


def analyze_directory(root_dir, extensions=('.py',)):
    """Analyze Python files in directory."""
    stats = {
        'total_files': 0,
        'total_lines': 0,
        'files_by_type': {},
    }
    
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            # Skip venv and cache
            if '.venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
            if 'egg-info' in str(file_path):
                continue
            
            lines = count_lines(file_path)
            stats['total_files'] += 1
            stats['total_lines'] += lines
            
            # Categorize by directory
            relative = file_path.relative_to(root_path)
            category = str(relative.parts[0]) if len(relative.parts) > 1 else 'root'
            
            if category not in stats['files_by_type']:
                stats['files_by_type'][category] = {'files': 0, 'lines': 0}
            
            stats['files_by_type'][category]['files'] += 1
            stats['files_by_type'][category]['lines'] += lines
    
    return stats


def print_stats():
    """Print project statistics."""
    root_dir = Path(__file__).parent.parent
    
    print("=" * 60)
    print("DML-PY Project Statistics")
    print("=" * 60)
    
    # Analyze code
    stats = analyze_directory(root_dir)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Python Files: {stats['total_files']}")
    print(f"  Total Lines of Code: {stats['total_lines']:,}")
    
    print(f"\n📁 Breakdown by Directory:")
    print(f"{'Category':<20} {'Files':<10} {'Lines':<10}")
    print("-" * 40)
    
    for category in sorted(stats['files_by_type'].keys()):
        data = stats['files_by_type'][category]
        print(f"{category:<20} {data['files']:<10} {data['lines']:<10,}")
    
    # Test statistics
    print(f"\n🧪 Testing:")
    test_dir = root_dir / 'tests'
    if test_dir.exists():
        test_files = list(test_dir.glob('test_*.py'))
        print(f"  Test Files: {len(test_files)}")
    
    # Example statistics
    print(f"\n📝 Examples:")
    examples_dir = root_dir / 'examples'
    if examples_dir.exists():
        example_files = list(examples_dir.glob('*.py'))
        print(f"  Example Scripts: {len(example_files)}")
    
    # Model statistics
    print(f"\n🏗️ Models:")
    models_dir = root_dir / 'pydml' / 'models' / 'cifar'
    if models_dir.exists():
        model_files = [f for f in models_dir.glob('*.py') if f.name != '__init__.py']
        print(f"  Model Architectures: {len(model_files)}")
        for model_file in sorted(model_files):
            print(f"    - {model_file.stem}")
    
    # Documentation
    print(f"\n📚 Documentation:")
    doc_files = ['README.md', 'PLAN.md', 'IMPLEMENTATION_SUMMARY.md', 'STATUS.md']
    for doc in doc_files:
        doc_path = root_dir / doc
        if doc_path.exists():
            lines = count_lines(doc_path)
            print(f"  {doc:<30} {lines:>6} lines")
    
    print("\n" + "=" * 60)
    print("✅ Phase 1 Implementation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    print_stats()
