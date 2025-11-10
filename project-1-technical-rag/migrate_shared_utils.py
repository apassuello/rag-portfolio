#!/usr/bin/env python3
"""
Migration script to update shared_utils imports to src.shared_utils
This consolidates the duplicated shared_utils directories into a single source.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Files to migrate (discovered from analysis)
FILES_TO_MIGRATE = [
    # Test files
    "tests/component/test_embeddings.py",
    "tests/component/test_pdf_parser.py",
    "tests/integration/test_integration.py",
    "tests/system/comprehensive_verification_test.py",
    "tests/tools/test_prompt_optimization.py",
    "tests/tools/test_prompt_simple.py",
    
    # Script files
    "scripts/debug_citation_issue.py",
    "scripts/debug_confidence_integration.py",
    "scripts/debug_confidence_issue.py",
    "scripts/debug_multi_doc_confidence.py",
    "scripts/debug_number_removal.py",
    "scripts/demos/demo_hybrid_search.py",
    "scripts/focused_chunk_analysis.py",
    
    # Core source files
    "src/basic_rag.py",
    "src/batch_document_processor.py",
    "src/rag_with_generation.py",
    
    # Internal shared_utils files
    "src/shared_utils/document_processing/hybrid_parser.py",
    "src/shared_utils/vector_stores/document_processing/hybrid_parser.py",
]

def create_backup_dir() -> Path:
    """Create backup directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"migration_backup_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    print(f"✓ Created backup directory: {backup_dir}")
    return backup_dir

def backup_file(file_path: str, backup_dir: Path) -> bool:
    """Backup a single file preserving directory structure."""
    try:
        source = Path(file_path)
        if not source.exists():
            print(f"  ⚠️  File not found: {file_path}")
            return False
        
        # Create subdirectories in backup
        dest = backup_dir / file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source, dest)
        print(f"  ✓ Backed up: {file_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to backup {file_path}: {e}")
        return False

def migrate_imports_in_file(file_path: str, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Update imports in a single file.
    Returns (success, number_of_changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match: from shared_utils (but not from src.shared_utils)
        pattern = r'^(from\s+)shared_utils(\s*import|\.)(?!.*from\s+src\.shared_utils)'
        replacement = r'\1src.shared_utils\2'
        
        # Count changes
        changes = len(re.findall(pattern, content, re.MULTILINE))
        
        if changes > 0:
            # Apply replacement
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"  ✓ Updated {file_path}: {changes} import(s) changed")
            else:
                print(f"  → Would update {file_path}: {changes} import(s)")
            
            return True, changes
        else:
            print(f"  - No changes needed in {file_path}")
            return True, 0
            
    except Exception as e:
        print(f"  ✗ Failed to process {file_path}: {e}")
        return False, 0

def run_migration(dry_run: bool = False):
    """Run the complete migration process."""
    print("\n" + "="*60)
    print("SHARED_UTILS IMPORT MIGRATION")
    print("="*60 + "\n")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    
    # Step 1: Create backup
    if not dry_run:
        print("Step 1: Creating backups...")
        backup_dir = create_backup_dir()
        
        backup_success = True
        for file_path in FILES_TO_MIGRATE:
            if not backup_file(file_path, backup_dir):
                backup_success = False
        
        if not backup_success:
            print("\n⚠️  Some files could not be backed up. Continue anyway? (y/n): ", end="")
            if input().lower() != 'y':
                print("Migration cancelled.")
                return
    
    # Step 2: Migrate imports
    print(f"\nStep 2: {'Analyzing' if dry_run else 'Migrating'} imports...")
    
    # Group files by category for better output
    categories = [
        ("Test Files", FILES_TO_MIGRATE[:6]),
        ("Script Files", FILES_TO_MIGRATE[6:13]),
        ("Core Source Files", FILES_TO_MIGRATE[13:16]),
        ("Internal shared_utils Files", FILES_TO_MIGRATE[16:])
    ]
    
    total_changes = 0
    failed_files = []
    
    for category_name, files in categories:
        print(f"\n{category_name}:")
        for file_path in files:
            success, changes = migrate_imports_in_file(file_path, dry_run)
            if success:
                total_changes += changes
            else:
                failed_files.append(file_path)
    
    # Step 3: Summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(FILES_TO_MIGRATE)}")
    print(f"Total imports {'would be' if dry_run else ''} updated: {total_changes}")
    
    if failed_files:
        print(f"\n⚠️  Failed to process {len(failed_files)} file(s):")
        for f in failed_files:
            print(f"  - {f}")
    else:
        print("\n✅ All files processed successfully!")
    
    if not dry_run and total_changes > 0:
        print("\nNext steps:")
        print("1. Run tests to verify the migration: pytest tests/")
        print("2. Test core functionality: python src/basic_rag.py")
        print("3. Once verified, remove ../shared_utils directory")
        print(f"\nBackup saved in: {backup_dir}")
        print("To restore: cp -r {backup_dir}/* .")

def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        print("Running in dry-run mode...")
        run_migration(dry_run=True)
    else:
        print("This will modify files. Run with --dry-run first to preview changes.")
        print("Continue with actual migration? (y/n): ", end="")
        if input().lower() == 'y':
            run_migration(dry_run=False)
        else:
            print("Migration cancelled. Run with --dry-run to preview changes.")

if __name__ == "__main__":
    main()