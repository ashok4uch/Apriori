
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import time
import requests
from requests.auth import HTTPBasicAuth
import argparse
import re
import boto3
from datetime import date
from aws_submit_batch_job import submit_and_wait
from launch_vision_qa import EC2
from multiprocessing import Process
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

parser = argparse.ArgumentParser(
    description='Pass any one environment against which automated test needs to be run <dev/uat/prd/>')
parser.add_argument('--env',
                    help='Pass any one environment against which automated test needs to be run <dev/uat/prd/>',
                    type=str, default=None)
parser.add_argument('--url', help='The URL which needs to be tested', type=str, default=None)
parser.add_argument('--vision_qa_user', help='User which can be used to test the URL', type=str, default=None)
parser.add_argument('--vision_qa_pwd', help='Password for the User', type=str, default=None)
parser.add_argument('--email_to', help='The Recipients to whom the email report need to be sent', type=str,
                    default=None)
parser.add_argument('--slpc_fi_perf_test_recipients', help='The Recipients to whom the email report need to be sent',
                    type=str, default=None)
parser.add_argument('--slpc_eq_perf_test_recipients', help='The Recipients to whom the email report need to be sent',
                    type=str, default=None)
parser.add_argument('--tags', help='BDD Tags to execute', type=str, default=None)
parser.add_argument('--threads', help='Number of parallel executions of pytest', type=str, default=None)
parser.add_argument('--browser_name', help='The browser on which the URL would be tested', type=str, default="chrome")
parser.add_argument('--bitbucket_build_id', help='The Bitbucket pipeline id', type=str, default="0")
parser.add_argument('--bitbucket_user', help='Bitbucket User', type=str, default=None)
parser.add_argument('--bitbucket_pwd', help='Bitbucket password', type=str, default=None)
parser.add_argument('--bitbucket_repo_owner', help='Bitbucket Repo owner', type=str, default=None)
parser.add_argument('--bitbucket_repo_slug', help='Bitbucket Repo name', type=str, default=None)
parser.add_argument('--bitbucket_repo_full_name', help='Bitbucket Repo full name', type=str, default=None)
parser.add_argument('--execution_env', help='Environment where the test is being executed', type=str, default="local")
parser.add_argument('--test_modules', help='comma separated list of values', type=str, default=None)
parser.add_argument('--test_dir', help='comma separated list of values', type=str, default="functional_tests")

args = parser.parse_args()

ENV = os.getenv('ENV', args.env)
URL = os.getenv('URL', args.url)
VISION_QA_USER = os.getenv('VISION_QA_USER', args.vision_qa_user)
VISION_QA_PWD = os.getenv('VISION_QA_PWD',
                          args.vision_qa_pwd)  # pass a raw string so that any special characters get interpreted as literals
EMAIL_RECIPIENTS = os.getenv('EMAILTO', args.email_to)
SLPC_FI_PERF_TEST_EMAIL_RECIPIENTS = os.getenv('SLPC_FI_PERF_TEST_EMAIL_RECIPIENTS', args.slpc_fi_perf_test_recipients)
SLPC_EQ_PERF_TEST_EMAIL_RECIPIENTS = os.getenv('SLPC_EQ_PERF_TEST_EMAIL_RECIPIENTS', args.slpc_eq_perf_test_recipients)
BROWSER_NAME = os.getenv('BROWSER_NAME', args.browser_name)
THREADS = os.getenv('THREADS', args.threads)
TAGS = os.getenv('TAGS', args.tags)
BITBUCKET_BUILD_ID = os.getenv('BUILD_ID', args.bitbucket_build_id)

BITBUCKET_USERNAME = os.getenv('DSO_BITBUCKET_USERNAME', args.bitbucket_user)
BITBUCKET_APP_PASSWD = os.getenv('DSO_BITBUCKET_APP_PASSWD', args.bitbucket_pwd)
BITBUCKET_REPO_OWNER = os.getenv('BITBUCKET_REPO_OWNER', args.bitbucket_repo_owner)
BITBUCKET_REPO_SLUG = os.getenv('BITBUCKET_REPO_SLUG', args.bitbucket_repo_slug)
BITBUCKET_REPO_FULL_NAME = os.getenv('BITBUCKET_REPO_FULL_NAME', args.bitbucket_repo_full_name)
EXECUTION_ENV = os.getenv('EXECUTION_ENV', args.execution_env).lower()
TEST_MODULES = os.getenv('TEST_MODULES', args.test_modules)
TEST_DIR = os.getenv('TEST_DIR', args.test_dir)

if EXECUTION_ENV not in ["local", "aws_batch", "aws_ec2"]:
    raise ValueError("Invalid Execution Environment.")

"""
ENV="uat"
BROWSER_NAME="chrome"
URL="https://vision.uat.invesco.com/"
UID="IVZ-VisionQA@amvescap.net"
PWD="1nve$c0V1$10n"
EMAIL_RECIPIENTS=None
TAGS=None
THREADS="4"

BITBUCKET_BUILD_ID="0"
BITBUCKET_USERNAME=None
BITBUCKET_APP_PASSWD=None
BITBUCKET_REPO_OWNER=None
BITBUCKET_REPO_SLUG=None
BITBUCKET_REPO_FULL_NAME=""
"""

PYTHONEXE = sys.executable
if TEST_MODULES:
    TEST_MODULES_LIST = TEST_MODULES.split(",")
else:
    TEST_MODULES_LIST = []

DATETIMESTAMP = time.strftime("%Y-%m-%d", time.localtime()) + '_' + time.strftime("%H-%M-%S", time.localtime())
S3_FOLDER = 'test_artifacts/' + str(ENV) + '/' + str(BITBUCKET_BUILD_ID) + '_' + str(DATETIMESTAMP)

S3_BUCKET = 'ivz-dev-0106-vision-qa-ue1'

ARTIFACT_LINK = "s3://" + S3_BUCKET + "/" + S3_FOLDER
REPORT_DIR = str((Path(__file__).parent).joinpath("results"))
os.makedirs(REPORT_DIR, exist_ok=True)

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
qa_s3_bucket = s3_resource.Bucket(S3_BUCKET)

#########################################
# Step 1 - Fetch Feature Files from Zephyr
#########################################
logging.info("Fetching Test Cases from Zephyr \r\n")
start = time.perf_counter()
print(
    "Executing " + "\"" + PYTHONEXE + "\" zephyr_test_cases.py --report_dir=" + REPORT_DIR + " --S3_FOLDER=" + S3_FOLDER + " --S3_BUCKET=" + S3_BUCKET + " --execution_env=" + EXECUTION_ENV + " --test_dir=" + TEST_DIR)
status = os.system(
    "\"" + PYTHONEXE + "\" zephyr_test_cases.py --report_dir=" + REPORT_DIR + " --S3_FOLDER=" + S3_FOLDER + " --S3_BUCKET=" + S3_BUCKET + " --execution_env=" + EXECUTION_ENV + " --test_dir=" + TEST_DIR)
status = os.waitstatus_to_exitcode(status) if 'linux' in sys.platform else status
if status != 0:
    exit(status)

end = time.perf_counter()
logging.info("Time taken to fetch Test Cases from Zephyr: " + str(end - start) + " sec")

##############################################
# Step 2 - Run pytest
##############################################
start = time.perf_counter()
processes = []
# If the job is being run on AWS, then run pytest on seperate containers
if EXECUTION_ENV == "aws_batch":
    if TEST_MODULES:
        TEST_MODULE_COUNT = len(TEST_MODULES.split(","))
    else:
        TEST_MODULE_COUNT = len([f for f in os.listdir(os.path.join('.', TEST_DIR)) if
                                 os.path.isfile(os.path.join('.', TEST_DIR, f)) and re.search('^test_.*.py$', f)])

    if TEST_MODULE_COUNT > 0:
        p = Process(target=submit_and_wait, args=(BITBUCKET_BUILD_ID,
                                                  TEST_MODULE_COUNT,
                                                  TEST_MODULES,
                                                  TEST_DIR,
                                                  ENV,
                                                  URL,
                                                  VISION_QA_USER,
                                                  VISION_QA_PWD,
                                                  BROWSER_NAME,
                                                  S3_FOLDER,
                                                  S3_BUCKET,))
        p.start()
        processes.append(p)
        processes.append(p)

    for p in processes:
        p.join()

# elif EXECUTION_ENV in ["aws_ec2"]:
#     print("Spawn multiple Ec2 machines and run the tests in parallel")
#     #TEST_DIR ='bdd_functional_tests'
#     ec2Instances = {}
#     ec2 = EC2()

#     if not TEST_MODULES:
#         TEST_MODULES = sorted([f for f in os.listdir(os.path.join('.', TEST_DIR)) if
#                                os.path.isfile(os.path.join('.', TEST_DIR,f)) and re.search('^test_.*.py$',f)])

#     for i,TEST_MODULE in enumerate(TEST_MODULES):
#         instance = ec2.launch_ec2_instance(i,TEST_MODULE,ENV,URL,VISION_QA_USER,VISION_QA_PWD,EMAIL_RECIPIENTS,BROWSER_NAME,BITBUCKET_BUILD_ID,EXECUTION_ENV)
#         ec2Instances[TEST_MODULE]=instance.id

#     status = ec2.ec2_instance_status(ec2Instances)
#     if status == "max_run_time":
#         print("Max. run time exceeded. Aborting process.")
#         exit(1)

#     ec2.terminate_ec2_instance(ec2Instances)
# PYTEST_CSV_OPTIONS = ' --csv=' + REPORT_DIR + os.sep + 'pytest_bdd_results.csv --csv-delimiter "|" --csv-columns module,class,name,status,duration,markers_as_columns'
# PYTEST_HTML_OPTIONS = ' --html=' + REPORT_DIR + os.sep + 'pytest_bdd_results.html --self-contained-html'
# PLAYWRIGHT_OPTIONS = ' --browser=chromium --browser-channel=chrome --headed --video=retain-on-failure --screenshot=only-on-failure --tracing=off'
# PLAYWRIGHT_OUTPUT_DIR = ' --output=test-results'
# PYTEST_BASE_COMMAND = 'pytest -W ignore -ra --tb=line ' + PLAYWRIGHT_OPTIONS + PLAYWRIGHT_OUTPUT_DIR + PYTEST_CSV_OPTIONS + PYTEST_HTML_OPTIONS + ' --url '+ URL +' --auth-id ' + VISION_QA_USER + ' --auth-pwd ' + VISION_QA_PWD + ' '

# print('Executing ' + PYTEST_BASE_COMMAND + ".")
# status = os.system('cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + '.')
# if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), TEST_DIR,"test-results")):
#    files = os.listdir(os.path.join('.', TEST_DIR,"test-results"))
#    for file in files:
#        if file.endswith(".webm"):
#            shutil.copy2(os.path.join(os.path.dirname(os.path.realpath(__file__)), TEST_DIR,"test-results", file), REPORT_DIR)

# If the job is being run locally, just run the pytest commands
elif EXECUTION_ENV in ["local", "aws_ec2"] and "bdd" in TEST_DIR or "re_write" in TEST_DIR:
    # TEST_DIR ='bdd_functional_tests'

    PYTEST_CSV_OPTIONS = ' --csv=' + REPORT_DIR + os.sep + 'pytest_bdd_results.csv --csv-delimiter "|" --csv-columns module,class,name,status,duration,markers_as_columns'
    PYTEST_HTML_OPTIONS = ' --html=' + REPORT_DIR + os.sep + 'pytest_bdd_results.html --self-contained-html'
    PLAYWRIGHT_OPTIONS = ' --browser=chromium --browser-channel=chromium --headed --video=retain-on-failure --screenshot=only-on-failure --tracing=off'
    PLAYWRIGHT_OUTPUT_DIR = ' --output=test-results'
    if THREADS:
        PYTEST_PARALLEL_OPTIONS = ' --dist=loadfile -n ' + THREADS
    else:
        PYTEST_PARALLEL_OPTIONS = ''

    PYTEST_BASE_COMMAND = 'pytest -W ignore -ra --tb=line ' + PYTEST_PARALLEL_OPTIONS + str(PLAYWRIGHT_OPTIONS) + str(
        PLAYWRIGHT_OUTPUT_DIR) + str(PYTEST_CSV_OPTIONS) + str(PYTEST_HTML_OPTIONS) + ' --url ' + str(
        URL) + ' --auth-id ' + str(VISION_QA_USER) + ' --auth-pwd ' + str(VISION_QA_PWD) + ' '
    if len(TEST_MODULES_LIST) > 0:
        logging.info('Executing ' + PYTEST_BASE_COMMAND + " ".join(TEST_MODULES_LIST))
        status = os.system('cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + " ".join(TEST_MODULES_LIST))
    else:
        logging.info('Executing ' + PYTEST_BASE_COMMAND + ".")
        status = os.system('cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + '.')

    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), TEST_DIR, "test-results")):
        files = os.listdir(os.path.join('.', TEST_DIR, "test-results"))
        for file in files:
            if file.endswith(".webm"):
                shutil.copy2(os.path.join(os.path.dirname(os.path.realpath(__file__)), TEST_DIR, "test-results", file),
                             REPORT_DIR)

    if "aws" in EXECUTION_ENV:
        print("uploading artifacts to S3")
        s3 = boto3.client('s3')
        for f in os.listdir(REPORT_DIR):
            with open(REPORT_DIR + os.sep + f, "rb") as fb:
                s3.upload_fileobj(fb, S3_BUCKET, S3_FOLDER + '/' + f)


elif EXECUTION_ENV in ["local", "aws_ec2"] and "bdd" not in TEST_DIR or "re_write" not in TEST_DIR:
    PYTEST_CSV_OPTIONS = ' --csv=' + REPORT_DIR + os.sep + 'pytest_results.csv --csv-delimiter "|" --csv-columns module,class,name,status,duration,properties_as_columns'
    PYTEST_HTML_OPTIONS = ' --html=' + REPORT_DIR + os.sep + 'pytest_results.html --self-contained-html'

    if THREADS:
        PYTEST_PARALLEL_OPTIONS = ' --dist=loadfile -n ' + THREADS
    else:
        PYTEST_PARALLEL_OPTIONS = ''

    PYTEST_BASE_COMMAND = 'pytest -W ignore -ra --tb=line' + PYTEST_PARALLEL_OPTIONS + PYTEST_CSV_OPTIONS + PYTEST_HTML_OPTIONS + ' --headless=false' + ' --url ' + URL + ' --auth_id ' + VISION_QA_USER + ' --auth_pwd ' + VISION_QA_PWD + ' --execution_env local  --report_home ' + REPORT_DIR + ' '

    if len(TEST_MODULES_LIST) > 0:
        logging.info('Executing ' + 'cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + " ".join(TEST_MODULES_LIST))
        status = os.system('cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + " ".join(TEST_MODULES_LIST))
    else:
        logging.info('Executing ' + 'cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + ".")
        status = os.system('cd ' + TEST_DIR + ' && ' + PYTEST_BASE_COMMAND + ".")
    status = os.waitstatus_to_exitcode(status) if 'linux' in sys.platform else status
    logging.info("Pytest execution finished with an exit code of " + str(status))

    if "aws" in EXECUTION_ENV:
        print("uploading artifacts to S3")
        s3 = boto3.client('s3')
        for f in os.listdir(REPORT_DIR):
            with open(REPORT_DIR + os.sep + f, "rb") as fb:
                s3.upload_fileobj(fb, S3_BUCKET, S3_FOLDER + '/' + f)

end = time.perf_counter()
logging.info("Time taken to run all the tests : " + str(end - start) + " sec")

#########################################
# Step 3 - Prepare Results
#########################################
start = time.perf_counter()
today = date.today()
value = today.strftime("%m/%d/%Y")
pytestBDDResultsDF = pd.DataFrame(
    columns=["module", "class", "name", "status", "duration", "Automated", "parameter_id", "test_case_id"])
pytestResultsDF = pd.DataFrame(columns=["module", "class", "name", "status", "duration", "test_case_id"])
fi_perfResultsDF = pd.DataFrame(columns=["Model", "URL", "Model Date", "Action", "Fastest Time", value])
eq_perfResultsDF = pd.DataFrame(columns=["Model", "Action", "Benchmark", value])

if "aws_batch" == EXECUTION_ENV:
    logging.info("Downloading files from S3")
    for s3_object in qa_s3_bucket.objects.filter(Prefix=S3_FOLDER):
        # Need to split s3_object.key into path and file name, else it will give error file not found.
        path, filename = os.path.split(s3_object.key)
        if S3_FOLDER in path and str(filename).endswith("_bdd_results.csv"):
            tempDF = pd.read_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + filename, sep="|")
            pytestBDDResultsDF = pd.concat([pytestBDDResultsDF, tempDF])
        if S3_FOLDER in path and (str(filename).endswith("_results.csv") and "_bdd_" not in filename):
            tempDF = pd.read_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + filename, sep="|")
            pytestResultsDF = pd.concat([pytestResultsDF, tempDF])
        if S3_FOLDER in path and (str(filename).endswith("_fi_perf_results.csv") and "_bdd_" not in filename):
            fi_perf_DF = pd.read_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + filename, sep=",", dtype='object')
            fi_perfResultsDF = pd.concat([fi_perfResultsDF, fi_perf_DF])
        if S3_FOLDER in path and (str(filename).endswith("_eq_perf_results.csv") and "_bdd_" not in filename):
            eq_perf_DF = pd.read_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + filename, sep=",", dtype='object')
            eq_perfResultsDF = pd.concat([eq_perfResultsDF, eq_perf_DF])

    pytestBDDResultsDF.to_csv(REPORT_DIR + os.sep + "pytest_bdd_results.csv", sep="|")
    pytestResultsDF.to_csv(REPORT_DIR + os.sep + "pytest_results.csv", sep="|")
    fi_perfResultsDF.to_csv(REPORT_DIR + os.sep + "slpc_fi_performance_results.csv", index=False)
    eq_perfResultsDF.to_csv(REPORT_DIR + os.sep + "slpc_eq_performance_results.csv", index=False)
    logging.info("Successfully downloaded files from S3")
elif EXECUTION_ENV in ["aws_ec2", "local"]:
    if os.path.exists(REPORT_DIR + os.sep + "pytest_bdd_results.csv"):
        pytestBDDResultsDF = pd.read_csv(REPORT_DIR + os.sep + "pytest_bdd_results.csv", sep="|")

    if os.path.exists(REPORT_DIR + os.sep + "pytest_results.csv"):
        pytestResultsDF = pd.read_csv(REPORT_DIR + os.sep + "pytest_results.csv", sep="|")

zephyrTestCasesDF = pd.read_csv(REPORT_DIR + os.sep + "zephyr_test_cases.csv", sep="|")
if any(TEST_DIR in str(x) for x in pytestResultsDF['module'].values):
    pytestResultsDF['module'] = pytestResultsDF['module'].apply(lambda x: str(x).replace(TEST_DIR + ".", ""))
pytestResultsDF['Script Reference'] = pytestResultsDF['module'] + '.' + pytestResultsDF['class'] + '.' + \
                                      pytestResultsDF['name']
pytestResultsDF['Script Reference'] = pytestResultsDF['Script Reference'].apply(lambda x: str(x).lower())

dfFilter = (zephyrTestCasesDF["ZEPHYR APPROVAL STATUS"] == "7 - Automation - Approved") | (
        zephyrTestCasesDF["ZEPHYR APPROVAL STATUS"] == "3 - Manual - Approved") | (
                   zephyrTestCasesDF["ZEPHYR APPROVAL STATUS"] == "4 - Manual - Refactor")
mergedDF_from_new_framework = pd.merge(zephyrTestCasesDF[dfFilter], pytestBDDResultsDF, how='left',
                                       left_on=['ZEPHYR TEST CASE ID', 'PARAMETER ID'],
                                       right_on=["test_case_id", "parameter_id"])
mergedDF_from_old_framework = pd.merge(zephyrTestCasesDF[(zephyrTestCasesDF[
                                                              "ZEPHYR APPROVAL STATUS"] == "9 - Automation - Approved - BDD in Progress") | (
                                                                 zephyrTestCasesDF[
                                                                     "ZEPHYR APPROVAL STATUS"] == "8 - Automation - Refactor")],
                                       pytestResultsDF, how='left', left_on=['Script Reference'],
                                       right_on=['Script Reference'])

mergedDF = pd.concat([mergedDF_from_new_framework, mergedDF_from_old_framework], axis=0)
mergedDF = mergedDF[
    ["ZEPHYR TEST CASE ID", "ZEPHYR TEST NAME", 'VISION CAPABILITY', 'ZEPHYR FOLDER', 'ZEPHYR APPROVAL STATUS',
     'ZEPHYR TEST CASE OWNER', 'ZEPHYR TEST CASE PRIORITY', 'ZEPHYR EXECUTION TYPE', "PARAMETER ID", "status",
     "duration", 'Script Reference']]
mergedDF.rename(columns={'status': 'TEST EXECUTION STATUS', 'duration': 'TEST EXECUTION TIME'}, inplace=True)
mergedDF.to_csv(REPORT_DIR + os.sep + "execution_results.csv", sep="|", index=False)

end = time.perf_counter()
logging.info("Time taken to prepare execution results: " + str(end - start) + " sec")

#########################################
# Step 4 - Upload Results to Zephyr
#########################################
# Don't upload results to Zephyr for local executions and performance tests
if "aws" in EXECUTION_ENV and not any("performance_testing.py" in x for x in TEST_MODULES_LIST):
    # if int(BITBUCKET_BUILD_ID)>0:
    logging.info("Uploading Results to Zephyr")
    start = time.perf_counter()

    args = ["--report_dir=" + REPORT_DIR]
    args.extend(["--env=" + ENV])
    args.extend(["--iteration=" + BITBUCKET_BUILD_ID])
    args.extend(["--datetimestamp=" + DATETIMESTAMP])
    status = os.system("\"" + PYTHONEXE + "\" zephyr_test_executions.py " + " ".join(args))
    status = os.waitstatus_to_exitcode(status) if 'linux' in sys.platform else status
    if status != 0:
        print("Test Execution upload to Zephyr failed.")
        exit(status)

    end = time.perf_counter()
    logging.info("Time taken to Upload execution results to Zephyr: " + str(end - start) + " sec")

#########################################
# Step 5 - Upload Artifacts to AWS S3
#########################################
if "aws" in EXECUTION_ENV:
    logging.info("Uploading test artifacts to S3 Bucket")
    # Read the Execution Results file again, as the previous step updates the file
    mergedDF = pd.read_csv(REPORT_DIR + os.sep + "execution_results.csv", sep="|")
    mergedDF.to_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + "execution_results.csv", sep="|", index=False)
    if "aws_batch" == EXECUTION_ENV:
        pytestBDDResultsDF.to_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + "pytest_bdd_results.csv", sep="|",
                                  index=False)
        pytestResultsDF.to_csv("s3://" + S3_BUCKET + "/" + S3_FOLDER + "/" + "pytest_results.csv", sep="|", index=False)
    logging.info("Finished uploading test artifacts to S3 Bucket")

# Deprecated code to upload results to Bitbucket downloads
"""if BITBUCKET_USERNAME and BITBUCKET_APP_PASSWD and BITBUCKET_REPO_OWNER and BITBUCKET_REPO_SLUG:
    print("Uploading test execution artifacts to the Bitbucket's Downloads folder")
    start = time.perf_counter()
    ARTIFACT_NAME = BITBUCKET_BUILD_ID + "_artifacts_" + DATETIMESTAMP
    shutil.make_archive(ARTIFACT_NAME, 'zip', REPORT_DIR)
    url = f'https://api.bitbucket.org/2.0/repositories/{BITBUCKET_REPO_OWNER}/{BITBUCKET_REPO_SLUG}/downloads'
    files = {'files': open(ARTIFACT_NAME + '.zip', 'rb'), }

    r = requests.post(url, auth=HTTPBasicAuth(BITBUCKET_USERNAME, BITBUCKET_APP_PASSWD), files=files)
    if r.status_code != 201:
        print("Error uploading artifacts to Bitbucket : " + r.text)
    else:
        print("Uploaded Test artifacts to Bitbucket successfully")

    end = time.perf_counter()
    print("Time taken to upload artifacts to Bitbucket : " + str(end - start) + " sec")
"""

#########################################
# Step 6 - Send email report
#########################################
if EMAIL_RECIPIENTS and not any("performance_testing.py" in x for x in TEST_MODULES_LIST):
    logging.info("Sending Email Report")
    start = time.perf_counter()
    notifyArgs = []
    notifyArgs.append("--env=" + ENV)
    notifyArgs.append("--url=" + URL)
    notifyArgs.append("--browser=" + BROWSER_NAME)
    notifyArgs.append("--report_dir=" + REPORT_DIR)
    notifyArgs.append('--emailTo="' + EMAIL_RECIPIENTS + '"')
    notifyArgs.append('--emailCc=""')
    notifyArgs.append('--execution_env=' + EXECUTION_ENV)
    notifyArgs.append("--artifact_url=" + ARTIFACT_LINK)
    if TEST_DIR == "functional_tests":
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_notification.py ' + ' '.join(notifyArgs) + '"')
    elif TEST_DIR == "bdd_functional_tests":
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_bdd_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_bdd_notification.py ' + ' '.join(notifyArgs) + '"')
    elif TEST_DIR == "vision_re_write":
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_bdd_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_bdd_notification.py ' + ' '.join(notifyArgs) + '"')
    # if TEST_DIR == "bdd_functional_tests" and os.path.exists(
    #     REPORT_DIR + os.path.sep + 'spa_slides_loading_results.csv'):
    #     status = os.system('""' + PYTHONEXE + '" send_pdf_slide_notification.py ' + ' '.join(notifyArgs) + '"')
    #     status = os.waitstatus_to_exitcode(status)

    if status != 0:
        print("Error sending email report")
        exit(status)

    end = time.perf_counter()
    logging.info("Time taken to send email notification: " + str(end - start) + " sec")

if SLPC_FI_PERF_TEST_EMAIL_RECIPIENTS and os.path.exists(
        REPORT_DIR + os.path.sep + 'slpc_fi_performance_results.csv') and (
        any("_slpc_fi_performance_testing.py" in x for x in TEST_MODULES_LIST) and len(TEST_MODULES_LIST) > 0):
    start = time.perf_counter()
    notifyArgs = []
    notifyArgs.append("--env=" + ENV)
    notifyArgs.append("--url=" + URL)
    notifyArgs.append("--browser=" + BROWSER_NAME)
    notifyArgs.append("--report_dir=" + REPORT_DIR)
    notifyArgs.append('--emailTo="' + SLPC_FI_PERF_TEST_EMAIL_RECIPIENTS + '"')
    notifyArgs.append('--emailCc=""')
    notifyArgs.append("--artifact_url=" + ARTIFACT_LINK)
    notifyArgs.append("--report_type=FI")
    if EXECUTION_ENV in ['aws_batch']:
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_perf_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_perf_notification.py ' + ' '.join(notifyArgs) + '"')
    elif EXECUTION_ENV in ['aws_ec2', 'local']:
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_perf_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_perf_notification.py ' + ' '.join(notifyArgs) + '"')
    if status != 0:
        logging.info("Error sending email report")
        exit(status)

    end = time.perf_counter()
    logging.info("Time taken to send email notification: " + str(end - start) + " sec")

if SLPC_EQ_PERF_TEST_EMAIL_RECIPIENTS and os.path.exists(
        REPORT_DIR + os.path.sep + 'slpc_eq_performance_results.csv') and (
        any("_slpc_eq_performance_testing.py" in x for x in TEST_MODULES_LIST) and len(TEST_MODULES_LIST) > 0):
    start = time.perf_counter()
    notifyArgs = []
    notifyArgs.append("--env=" + ENV)
    notifyArgs.append("--url=" + URL)
    notifyArgs.append("--browser=" + BROWSER_NAME)
    notifyArgs.append("--report_dir=" + REPORT_DIR)
    notifyArgs.append('--emailTo="' + SLPC_EQ_PERF_TEST_EMAIL_RECIPIENTS + '"')
    notifyArgs.append('--emailCc=""')
    notifyArgs.append("--artifact_url=" + ARTIFACT_LINK)
    notifyArgs.append("--report_type=EQ")
    if EXECUTION_ENV in ['aws_batch']:
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_perf_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_perf_notification.py ' + ' '.join(notifyArgs) + '"')
    elif EXECUTION_ENV in ['aws_ec2', 'local']:
        if 'linux' in sys.platform or 'cygwin' in sys.platform:
            status = os.system(PYTHONEXE + ' send_perf_notification.py ' + ' '.join(notifyArgs))
            status = os.waitstatus_to_exitcode(status)
        else:
            status = os.system('""' + PYTHONEXE + '" send_perf_notification.py ' + ' '.join(notifyArgs) + '"')
    if status != 0:
        logging.info("Error sending email report")
        exit(status)

    end = time.perf_counter()
    logging.info("Time taken to send email notification: " + str(end - start) + " sec")
