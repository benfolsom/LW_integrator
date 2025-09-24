"""
Comprehensive test runner for the LW integrator test suite.

This script provides convenient functions to run different categories of tests
and generate test reports with performance metrics.

Author: Ben Folsom
Date: 2025-09-18
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class LWTestRunner:
    """Comprehensive test runner for LW integrator tests."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_dir = Path(__file__).parent
        self.results = {}

    def run_pytest(self, args: List[str], test_name: str) -> Dict:
        """Run pytest with specified arguments and capture results."""

        cmd = ["python", "-m", "pytest"] + args

        if self.verbose:
            print(f"\\n🧪 Running {test_name}...")
            print(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time

            success = result.returncode == 0

            if self.verbose:
                if success:
                    print(f"✅ {test_name} passed in {duration:.2f}s")
                else:
                    print(f"❌ {test_name} failed in {duration:.2f}s")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")

            return {
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} timed out after 5 minutes")
            return {
                "success": False,
                "duration": 300.0,
                "stdout": "",
                "stderr": "Test timed out",
                "returncode": -1,
            }
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            return {
                "success": False,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": str(e),
                "returncode": -2,
            }

    def run_unit_tests(self) -> Dict:
        """Run all unit tests."""
        args = ["unit/", "-v", "-m", "unit", "--tb=short"]
        return self.run_pytest(args, "Unit Tests")

    def run_integration_tests(self) -> Dict:
        """Run integration tests."""
        args = ["integration/", "-v", "-m", "integration", "--tb=short"]
        return self.run_pytest(args, "Integration Tests")

    def run_performance_tests(self) -> Dict:
        """Run performance and scaling tests."""
        args = [
            "benchmarks/test_performance_scaling.py",
            "-v",
            "-m",
            "performance",
            "--tb=short",
        ]
        return self.run_pytest(args, "Performance Tests")

    def run_multi_species_tests(self) -> Dict:
        """Run multi-species and radiation reaction tests."""
        args = [
            "benchmarks/test_multi_species_validation.py",
            "-v",
            "-m",
            "physics",
            "--tb=short",
        ]
        return self.run_pytest(args, "Multi-Species Tests")

    def run_physics_validation(self) -> Dict:
        """Run all physics validation tests."""
        args = ["-v", "-m", "physics", "--tb=short"]
        return self.run_pytest(args, "Physics Validation")

    def run_fast_tests(self) -> Dict:
        """Run fast tests only (excludes slow and performance tests)."""
        args = ["-v", "-m", "not slow and not performance", "--tb=short"]
        return self.run_pytest(args, "Fast Tests")

    def run_slow_tests(self) -> Dict:
        """Run slow tests only."""
        args = ["-v", "-m", "slow", "--tb=short"]
        return self.run_pytest(args, "Slow Tests")

    def run_all_tests(self) -> Dict:
        """Run the complete test suite."""
        args = ["-v", "--tb=short", "--durations=10"]  # Show 10 slowest tests
        return self.run_pytest(args, "Complete Test Suite")

    def run_specific_test(
        self, test_file: str, test_function: Optional[str] = None
    ) -> Dict:
        """Run a specific test file or function."""
        args = [test_file, "-v", "--tb=short"]

        if test_function:
            args[0] += f"::{test_function}"

        test_name = f"Specific Test: {test_file}"
        if test_function:
            test_name += f"::{test_function}"

        return self.run_pytest(args, test_name)

    def generate_report(self):
        """Generate a summary report of all test results."""

        print("\\n" + "=" * 60)
        print("🧪 LW INTEGRATOR TEST SUITE REPORT")
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        failed_tests = total_tests - passed_tests

        total_time = sum(result["duration"] for result in self.results.values())

        print("\\n📊 SUMMARY:")
        print(f"   Total test categories: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Success rate: {100 * passed_tests / total_tests:.1f}%")

        print("\\n⏱️  TIMING BREAKDOWN:")
        for test_name, result in self.results.items():
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {test_name}: {result['duration']:.2f}s")

        if failed_tests > 0:
            print("\\n❌ FAILED TESTS:")
            for test_name, result in self.results.items():
                if not result["success"]:
                    print(f"   {test_name}:")
                    print(f"     Return code: {result['returncode']}")
                    if result["stderr"]:
                        print(f"     Error: {result['stderr'][:200]}...")

        print("\\n" + "=" * 60)

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests,
            "total_time": total_time,
        }


def main():
    """Main test runner function."""

    print("🚀 LW Integrator Test Suite Runner")
    print("Available test categories:")
    print("  1. Unit tests")
    print("  2. Integration tests")
    print("  3. Performance tests")
    print("  4. Multi-species tests")
    print("  5. Physics validation")
    print("  6. Fast tests only")
    print("  7. Slow tests only")
    print("  8. Complete test suite")
    print("  9. Custom test")

    runner = LWTestRunner(verbose=True)

    # Check for command line arguments
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\\nSelect test category (1-9): ").strip()

    # Run selected tests
    if choice == "1":
        runner.results["Unit Tests"] = runner.run_unit_tests()
    elif choice == "2":
        runner.results["Integration Tests"] = runner.run_integration_tests()
    elif choice == "3":
        runner.results["Performance Tests"] = runner.run_performance_tests()
    elif choice == "4":
        runner.results["Multi-Species Tests"] = runner.run_multi_species_tests()
    elif choice == "5":
        runner.results["Physics Validation"] = runner.run_physics_validation()
    elif choice == "6":
        runner.results["Fast Tests"] = runner.run_fast_tests()
    elif choice == "7":
        runner.results["Slow Tests"] = runner.run_slow_tests()
    elif choice == "8":
        runner.results["Complete Suite"] = runner.run_all_tests()
    elif choice == "9":
        test_file = input("Enter test file path: ").strip()
        test_func = input("Enter test function (optional): ").strip()
        test_func = test_func if test_func else None
        runner.results["Custom Test"] = runner.run_specific_test(test_file, test_func)
    elif choice.lower() in ["all", "complete"]:
        # Run comprehensive test suite
        print("\\n🎯 Running comprehensive test suite...")

        runner.results["Unit Tests"] = runner.run_unit_tests()
        runner.results["Integration Tests"] = runner.run_integration_tests()
        runner.results["Performance Tests"] = runner.run_performance_tests()
        runner.results["Multi-Species Tests"] = runner.run_multi_species_tests()
    else:
        print(f"Invalid choice: {choice}")
        return

    # Generate report
    summary = runner.generate_report()

    # Exit with appropriate code
    if summary["failed"] > 0:
        sys.exit(1)
    else:
        print("\\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
