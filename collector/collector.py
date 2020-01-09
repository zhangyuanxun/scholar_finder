from constants import *
import csv
from tqdm import tqdm
from utils import *
import json
import multiprocessing
import sys


def get_scholar_info_from_grant(row):
    # award id
    award_id = int(row[0])

    # get institutions
    institute = row[1].split('@')

    if len(institute) != 2:
        return None, None, None

    institute = institute[1].lower()

    # get scholar names
    first_name = row[2]
    last_name = row[3]

    if first_name == "None":
        return None, None, None

    name = first_name + ' ' + last_name
    name = name.lower()

    if name < 3:  # skip invalid name, the length smaller than 3
        return None, None, None

    return name, institute, award_id


def collect_google_profile(worker_idx, start_idx, num_collected):
    scholar_profile = {}

    with open(SCHOLAR_AWARDS, 'r') as fp:
        author_award = json.load(fp)

    scholar_award_num = {}
    for author in author_award.keys():
        scholar_award_num[author] = len(author_award[author])

    scholar_award_num = sorted(scholar_award_num.items(), key=lambda kv: kv[1], reverse=True)
    scholar_award_num = scholar_award_num[start_idx: start_idx + num_collected]
    print print_timestamp() + "worker-%d collect scholar from %d to %d" % (worker_idx, start_idx, start_idx + num_collected)
    cnt = 0
    for scholar in scholar_award_num:
        name, institute = scholar[0].split('@')
        profile = api_get_google_profile(name, institute)
        cnt += 1
        if cnt % 10 == 0:
            print print_timestamp() + "worker-%d: Finish %d scholar profile collecting" % (worker_idx, cnt)

        if profile is None:
            continue

        scholar_profile[name] = profile
        sys.stdout.flush()

    # save files
    file_name = DATASET_PROCESSED_FOLDER + "scholar_profile_" + str(worker_idx) + ".json"
    with open(file_name, 'w') as fp:
        json.dump(scholar_profile, fp)


def multiprocessing_collect_google_profile(num_workers=10, start=0, num_collected=100):
    jobs = []
    for word_idx in range(num_workers):
        start_idx = word_idx * num_collected + start
        p = multiprocessing.Process(target=collect_google_profile, args=(word_idx, start_idx, num_collected))
        jobs.append(p)
        p.start()


def collect_scholar_award():
    scholar_award = {}

    # check total number
    with open(RAW_NSF_GRANT_INVESTIGATORS, 'r') as csv_file:
        grant_info = csv.reader(csv_file, delimiter=',')
        headers = next(grant_info)

        num_line = 0
        for row in grant_info:
            num_line += 1

    with open(RAW_NSF_GRANT_INVESTIGATORS, 'r') as csv_file:
        grant_info = csv.reader(csv_file, delimiter=',')
        headers = next(grant_info)

        with tqdm(total=num_line, unit="docs", desc="collect_scholar_award") as p_bar:
            for row in grant_info:
                p_bar.update(1)

                name, institute, award_id = get_scholar_info_from_grant(row)

                if name is None:
                    continue
                k = name + '@' + institute
                if k in scholar_award:
                    scholar_award[k].append(award_id)
                else:
                    scholar_award[k] = [award_id]

    scholar_award = {k: list(set(v)) for k, v in scholar_award.items()}

    # save files
    with open(SCHOLAR_AWARDS, 'w') as fp:
        json.dump(scholar_award, fp)


def merge_files():
    scholar_profiles = {}
    for i in range(79):
        # open file
        file_name = DATASET_PROCESSED_FOLDER + "scholar_profile_" + str(i) + ".json"
        try:
            with open(file_name, 'r') as fp:
                d = json.load(fp)
                scholar_profiles.update(d)
        except Exception:
            continue

    # update some scholars who are missig
    profile = api_get_google_profile('trupti joshi', 'missouri.edu')
    scholar_profiles['trupti joshi'] = profile

    # save files
    with open(SCHOLAR_PROFILES, 'w') as fp:
        json.dump(scholar_profiles, fp)


if __name__ == "__main__":
    multiprocessing_collect_google_profile(num_workers=20, start=0, num_collected=500)
    merge_files()