#!/usr/bin/env python3

import subprocess
import os
import re
import sys
import json
import platform

element_counts = ["1000000", "10000000", "20000000", "40000000"]
perf_points = 1.25

# ---------- Platform Check ----------
IS_WINDOWS = platform.system() == "Windows"

# ---------- Command Wrapper ----------
def run_command(command, capture_output=False):
    if IS_WINDOWS:
        full_cmd = ["powershell", "-Command", command]
        return subprocess.run(full_cmd, shell=True, capture_output=capture_output)
    else:
        if os.environ.get("GRADING_TOKEN"):
            return subprocess.run(command, shell=True, capture_output=capture_output, user="nobody", env={})
        else:
            return subprocess.run(command, shell=True, capture_output=capture_output)

# ---------- Setup Logs ----------
subprocess.run("rm -rf logs/*", shell=True) if not IS_WINDOWS else subprocess.run("Remove-Item -Recurse -Force logs\\*", shell=True)

os.makedirs("logs/test", exist_ok=True)
os.makedirs("logs/ref", exist_ok=True)
if os.environ.get("GRADING_TOKEN") and not IS_WINDOWS:
    subprocess.run("chown -R nobody:nogroup logs", shell=True)

# ---------- Parse test type ----------
if len(sys.argv) != 2 or sys.argv[1] not in ["find_repeats", "scan"]:
    print("Usage: python3 checker.py <test>: test = scan, find_repeats")
    sys.exit(1)
else:
    test = sys.argv[1]
    print(f"Test: {test}")

print("\n--------------")
print("Running tests:")
print("--------------")

# ---------- Test Function ----------
def check_correctness(test, element_count):
    exe = ".\\cudaScan.exe" if IS_WINDOWS else "./cudaScan"
    log_file = f"logs/test/{test}_correctness_{element_count}.log"
    command = f'{exe} -m {test} -i random -n {element_count} > {log_file}'
    result = run_command(command)
    return result.returncode == 0

def get_time(command):
    result = run_command(command, capture_output=True)
    output = result.stdout.decode()
    match = re.search(r"\d+(\.\d+)?", output)
    return float(match.group()) if match else None

# ---------- Run Tests ----------
def run_tests():
    correct = {}
    your_times = {}
    fast_times = {}

    for element_count in element_counts:
        print(f"\nElement Count: {element_count}")

        # Check correctness
        correct[element_count] = check_correctness(test, element_count)
        if correct[element_count]:
            print("Correctness passed!")
        else:
            print("Correctness failed")

        # Your time
        exe = ".\\cudaScan.exe" if IS_WINDOWS else "./cudaScan"
        grep = "findstr" if IS_WINDOWS else "grep"
        log_path = f"logs/test/{test}_time_{element_count}.log"
        student_cmd = f'{exe} -m {test} -i random -n {element_count} | Tee-Object -FilePath {log_path} | {grep} "Student GPU time:"' if IS_WINDOWS else f'{exe} -m {test} -i random -n {element_count} | tee {log_path} | grep "Student GPU time:"'
        your_times[element_count] = get_time(student_cmd)
        print(f"Student Time: {your_times[element_count]}")

        # Reference time
        ref_exe = ".\\cudaScan_ref_x86.exe" if IS_WINDOWS else "./cudaScan_ref_x86" if platform.machine() == "x86_64" else "./cudaScan_ref"
        ref_log = f"logs/ref/{test}_time_{element_count}.log"
        ref_cmd = f'{ref_exe} -m {test} -i random -n {element_count} | Tee-Object -FilePath {ref_log} | {grep} "Student GPU time:"' if IS_WINDOWS else f'{ref_exe} -m {test} -i random -n {element_count} | tee {ref_log} | grep "Student GPU time:"'
        fast_times[element_count] = get_time(ref_cmd)
        print(f"Ref Time: {fast_times[element_count]}")

    return correct, your_times, fast_times

# ---------- Scoring ----------
def calculate_scores(correct, your_times, fast_times):
    scores = []
    total_score = 0

    for element_count in element_counts:
        ref_time = fast_times[element_count]
        stu_time = your_times[element_count]

        if correct[element_count]:
            if stu_time <= 1.20 * ref_time:
                score = perf_points
            else:
                score = perf_points * (ref_time / stu_time)
        else:
            score = 0

        scores.append({
            "element_count": element_count,
            "correct": correct[element_count],
            "ref_time": ref_time,
            "stu_time": stu_time,
            "score": score,
        })
        total_score += score

    max_total_score = perf_points * len(element_counts)
    return scores, total_score, max_total_score

# ---------- Output ----------
def print_score_table(scores, total_score, max_total_score):
    print("\n-------------------------")
    print(f"{test.capitalize()} Score Table:")
    print("-------------------------")

    header = "| %-15s | %-15s | %-15s | %-15s |" % (
        "Element Count", "Ref Time", "Student Time", "Score")
    dashes = "-" * len(header)
    print(dashes)
    print(header)
    print(dashes)

    for score in scores:
        element_count = score["element_count"]
        ref_time = score["ref_time"]
        stu_time = score["stu_time"]
        score_value = score["score"]

        if not score["correct"]:
            stu_time = f"{stu_time} (F)"

        print(
            "| %-15s | %-15s | %-15s | %-15s |"
            % (element_count, ref_time, stu_time, score_value)
        )

    print(dashes)
    print(
        "| %-33s | %-15s | %-15s |"
        % ("", "Total score:", f"{total_score}/{max_total_score}")
    )
    print(dashes)

# ---------- Main ----------
correct, your_times, fast_times = run_tests()
scores, total_score, max_total_score = calculate_scores(correct, your_times, fast_times)

GRADING_TOKEN = os.environ.get("GRADING_TOKEN")
if not GRADING_TOKEN:
    print_score_table(scores, total_score, max_total_score)
else:
    scores = json.dumps(scores)
    print(f"{GRADING_TOKEN}{scores}")
