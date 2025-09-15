#!/usr/bin/env python3
"""
Comprehensive Test Categorization and Migration Script

This script analyzes all remaining test files and organizes them into appropriate
categories: active tests, archive (redundant/deprecated), and development.

Author: Ben Folsom (human oversight)
Date: 2025-09-15
"""

import shutil
from pathlib import Path

# COMPREHENSIVE TEST ANALYSIS
# ===========================

TEST_ANALYSIS = {
    # REDUNDANT TESTS - Multiple tests covering same functionality
    "REDUNDANT": {
        "test_radiation_reaction.py": {
            "reason": "Superseded by test_radiation_reaction_improved.py",
            "action": "archive/redundant",
            "better_version": "unit_tests/test_radiation_reaction.py"
        },
        "test_conductor_surface.py": {
            "reason": "Superseded by test_conductor_surface_improved.py", 
            "action": "archive/redundant",
            "better_version": "test_conductor_surface_improved.py"
        },
        "test_simple_performance.py": {
            "reason": "Basic version of test_performance_scaling.py",
            "action": "archive/redundant", 
            "better_version": "performance_tests/test_performance_scaling.py"
        },
        "test_legacy_threshold.py": {
            "reason": "Superseded by test_improved_threshold.py",
            "action": "archive/redundant",
            "better_version": "test_improved_threshold.py"
        },
        "debug_threshold.py": {
            "reason": "Debug version, functionality in improved tests",
            "action": "archive/redundant",
            "better_version": "test_improved_threshold.py"
        }
    },
    
    # DEPRECATED TESTS - Old approaches or broken tests
    "DEPRECATED": {
        "cross_verification.py": {
            "reason": "Failed interface compatibility, replaced by physics_consistent_comparison.py",
            "action": "archive/deprecated",
            "replacement": "physics_tests/test_physics_accuracy.py"
        },
        "final_state_comparison.py": {
            "reason": "Had unit conversion issues, fixed in physics_consistent_comparison.py", 
            "action": "archive/deprecated",
            "replacement": "physics_consistent_comparison.py"
        },
        "test_debug_two_particles.py": {
            "reason": "Debug script, not systematic test",
            "action": "archive/deprecated",
            "replacement": "development/debug_helpers.py"
        },
        "debug_legacy_init.py": {
            "reason": "Legacy debug script",
            "action": "archive/deprecated",
            "replacement": "utilities/"
        },
        "debug_legacy_trajectory.py": {
            "reason": "Legacy debug script", 
            "action": "archive/deprecated",
            "replacement": "utilities/"
        }
    },
    
    # INTEGRATION TESTS - Full system testing
    "INTEGRATION": {
        "basic_optimized_verification.py": {
            "reason": "Compares basic vs optimized integrators",
            "action": "integration_tests/test_basic_vs_optimized.py",
            "priority": "high"
        },
        "comprehensive_verification.py": {
            "reason": "Full system verification",
            "action": "integration_tests/test_comprehensive_verification.py", 
            "priority": "high"
        },
        "standalone_verification.py": {
            "reason": "Legacy vs modern comparison",
            "action": "integration_tests/test_legacy_compatibility.py",
            "priority": "medium"
        },
        "legacy_verification.py": {
            "reason": "Legacy system testing",
            "action": "integration_tests/test_legacy_verification.py",
            "priority": "medium"
        },
        "extended_verification.py": {
            "reason": "Extended verification scenarios",
            "action": "integration_tests/test_extended_scenarios.py",
            "priority": "medium"
        },
        "scalable_verification.py": {
            "reason": "Scalability testing",
            "action": "integration_tests/test_scalability.py",
            "priority": "medium"
        }
    },
    
    # PERFORMANCE TESTS - Benchmarking and optimization
    "PERFORMANCE": {
        "test_performance_scaling.py": {
            "reason": "Core performance scaling tests",
            "action": "performance_tests/test_scaling_benchmarks.py",
            "priority": "high"
        },
        "test_optimization_comparison.py": {
            "reason": "Optimization effectiveness testing",
            "action": "performance_tests/test_optimization_effectiveness.py",
            "priority": "high"
        },
        "test_large_scale_performance.py": {
            "reason": "Large-scale performance benchmarks",
            "action": "performance_tests/test_large_scale_benchmarks.py",
            "priority": "medium"
        },
        "test_final_performance.py": {
            "reason": "Final performance validation",
            "action": "performance_tests/test_final_validation.py",
            "priority": "medium"
        },
        "test_corrected_performance.py": {
            "reason": "Corrected performance tests",
            "action": "performance_tests/test_corrected_benchmarks.py",
            "priority": "low"
        }
    },
    
    # PHYSICS TESTS - Physics accuracy and validation
    "PHYSICS": {
        "physics_consistent_comparison.py": {
            "reason": "Core physics consistency validation",
            "action": "physics_tests/test_physics_consistency.py",
            "priority": "high"
        },
        "analyze_radiation_reaction.py": {
            "reason": "Radiation reaction physics analysis",
            "action": "physics_tests/test_radiation_analysis.py",
            "priority": "medium"
        },
        "radiation_reaction_summary.py": {
            "reason": "Radiation reaction summary analysis",
            "action": "physics_tests/test_radiation_summary.py",
            "priority": "low"
        },
        "validation_test.py": {
            "reason": "General physics validation",
            "action": "physics_tests/test_general_validation.py", 
            "priority": "medium"
        }
    },
    
    # SPECIALIZED TESTS - Specific features
    "SPECIALIZED": {
        "test_conductor_surface_improved.py": {
            "reason": "Conductor surface physics (improved version)",
            "action": "physics_tests/test_conductor_surface.py",
            "priority": "medium"
        },
        "test_radiation_reaction_improved.py": {
            "reason": "Improved radiation reaction tests (already converted)",
            "action": "unit_tests/test_radiation_reaction.py",
            "priority": "high",
            "status": "completed"
        },
        "test_char_time_scaling.py": {
            "reason": "Characteristic time scaling tests",
            "action": "physics_tests/test_char_time_scaling.py",
            "priority": "medium"
        },
        "test_extended_scaling.py": {
            "reason": "Extended scaling analysis",
            "action": "performance_tests/test_extended_scaling.py",
            "priority": "low"
        },
        "test_improved_threshold.py": {
            "reason": "Improved threshold testing",
            "action": "unit_tests/test_threshold_algorithms.py",
            "priority": "medium"
        },
        "test_radiation_reaction_debug.py": {
            "reason": "Radiation reaction debugging",
            "action": "development/test_radiation_debugging.py",
            "priority": "low"
        },
        "test_radiation_reaction_triggering.py": {
            "reason": "Radiation reaction trigger testing",
            "action": "unit_tests/test_radiation_triggers.py",
            "priority": "medium"
        },
        "test_refactored_package.py": {
            "reason": "Package refactoring validation",
            "action": "integration_tests/test_package_structure.py",
            "priority": "low"
        }
    },
    
    # UTILITY FILES - Support files
    "UTILITIES": {
        "test_adaptive_self_consistency.py": {
            "reason": "Already in place - adaptive integration test",
            "action": "keep in root (recently created)",
            "priority": "high",
            "status": "completed"
        },
        "test_organization_plan.py": {
            "reason": "Documentation file for organization",
            "action": "utilities/test_organization_plan.py",
            "priority": "low"
        },
        "migration_guide.py": {
            "reason": "Migration documentation",
            "action": "utilities/migration_guide.py", 
            "priority": "low"
        }
    }
}

def print_categorization_plan():
    """Print the complete categorization plan."""
    print("📋 COMPREHENSIVE TEST CATEGORIZATION PLAN")
    print("=" * 60)
    
    total_files = 0
    for category, tests in TEST_ANALYSIS.items():
        total_files += len(tests)
        
        print(f"\n📂 {category} ({len(tests)} files):")
        for test_file, details in tests.items():
            status = details.get('status', 'pending')
            priority = details.get('priority', 'medium')
            action = details['action']
            
            status_icon = "✅" if status == "completed" else "📋"
            priority_icon = {"high": "🔥", "medium": "⚠️", "low": "💡"}[priority]
            
            print(f"   {status_icon} {priority_icon} {test_file}")
            print(f"      → {action}")
            print(f"      💭 {details['reason']}")
            
            if 'better_version' in details:
                print(f"      🔄 Better version: {details['better_version']}")
            if 'replacement' in details:
                print(f"      🔄 Replacement: {details['replacement']}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files analyzed: {total_files}")
    
    redundant = len(TEST_ANALYSIS['REDUNDANT'])
    deprecated = len(TEST_ANALYSIS['DEPRECATED'])
    active = total_files - redundant - deprecated
    
    print(f"   🗑️  Archive (redundant): {redundant}")
    print(f"   🗂️  Archive (deprecated): {deprecated}")
    print(f"   ✅ Active tests: {active}")
    
    print(f"\n🎯 MIGRATION PRIORITIES:")
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for category, tests in TEST_ANALYSIS.items():
        for test_file, details in tests.items():
            if details.get('status') == 'completed':
                continue
            priority = details.get('priority', 'medium')
            if priority == 'high':
                high_priority.append((test_file, details['action']))
            elif priority == 'medium':
                medium_priority.append((test_file, details['action']))
            else:
                low_priority.append((test_file, details['action']))
    
    print(f"   🔥 High priority: {len(high_priority)} files")
    print(f"   ⚠️  Medium priority: {len(medium_priority)} files") 
    print(f"   💡 Low priority: {len(low_priority)} files")

def execute_migration():
    """Execute the migration plan."""
    print("\n🚀 EXECUTING MIGRATION PLAN...")
    
    # First, archive redundant and deprecated tests
    print("\n📦 Archiving redundant tests...")
    for test_file, details in TEST_ANALYSIS['REDUNDANT'].items():
        src = Path(test_file)
        if src.exists():
            dst = Path("archive/redundant") / test_file
            print(f"   🗑️  {test_file} → {dst}")
            # shutil.move(str(src), str(dst))  # Commented out for safety
        else:
            print(f"   ⚠️  {test_file} not found")
    
    print("\n📦 Archiving deprecated tests...")
    for test_file, details in TEST_ANALYSIS['DEPRECATED'].items():
        src = Path(test_file)
        if src.exists():
            dst = Path("archive/deprecated") / test_file
            print(f"   🗂️  {test_file} → {dst}")
            # shutil.move(str(src), str(dst))  # Commented out for safety
        else:
            print(f"   ⚠️  {test_file} not found")
    
    # Move active tests to appropriate directories
    print("\n📁 Moving active tests...")
    
    categories_to_move = ['INTEGRATION', 'PERFORMANCE', 'PHYSICS', 'SPECIALIZED', 'UTILITIES']
    for category in categories_to_move:
        if category in TEST_ANALYSIS:
            print(f"\n   📂 {category}:")
            for test_file, details in TEST_ANALYSIS[category].items():
                if details.get('status') == 'completed':
                    print(f"      ✅ {test_file} (already completed)")
                    continue
                    
                src = Path(test_file)
                dst_path = details['action']
                
                if src.exists():
                    print(f"      📋 {test_file} → {dst_path}")
                    # Create the destination directory if needed
                    dst = Path(dst_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    # shutil.move(str(src), str(dst))  # Commented out for safety
                else:
                    print(f"      ⚠️  {test_file} not found")

if __name__ == "__main__":
    print_categorization_plan()
    print("\n" + "="*60)
    execute_migration()
    
    print("\n✨ MIGRATION PLAN COMPLETE!")
    print("   📝 Review the plan above")
    print("   🔧 Uncomment shutil.move() calls to execute")
    print("   🧪 Convert moved tests to pytest format")
    print("   🗑️  Clean up archive periodically")