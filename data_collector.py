from constants import *
import csv
from tqdm import tqdm
from utils import *
import json
import multiprocessing
import sys


def collect_nsf_abstract():
    with open(KNOWLEDGE_BOW, 'r') as fp:
        scholar_profiles = json.load(fp)

    with open(SCHOLAR_GRANTS, 'r') as fp:
        scholar_awards = json.load(fp)

    scholar_names = scholar_profiles['names']
    scholar_names_dic = dict.fromkeys(scholar_names)

    award_ids = []
    for scholar in scholar_awards.keys():
        name = scholar.split('@')[0]

        if name in scholar_names_dic:
            award_ids += scholar_awards[scholar]

    print len(award_ids)
    print len(list(set(award_ids)))

    award_dic = dict.fromkeys(award_ids, 0)
    nsf_abstract_by_id = {}

    with open(RAW_NSF_GRANT_INFO, 'rb') as csv_file:
        grant_info = csv.reader(csv_file, delimiter=',')
        headers = next(grant_info)

        for row in grant_info:
            try:
                award_id = int(row[2])

                if award_id in award_dic:
                    abstract = row[0]
                    nsf_abstract_by_id[award_id] = abstract
            except Exception:
                continue

    with open(NSF_ABSTRACT_BY_ID, 'w') as fp:
        json.dump(nsf_abstract_by_id, fp)


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


def collect_scholar_publications():
    scholar_profile = {}

    with open(SCHOLAR_GRANTS, 'r') as fp:
        author_award = json.load(fp)

    scholar_award_num = {}
    for author in author_award.keys():
        scholar_award_num[author] = len(author_award[author])

    scholar_award_num = sorted(scholar_award_num.items(), key=lambda kv: kv[1], reverse=True)
    cnt = 0
    for scholar in scholar_award_num:
        name, institute = scholar[0].split('@')
        profile = api_get_google_profile(name, institute)

        if profile is None:
            continue

        scholar_profile[name] = profile
        cnt += 1
        if cnt % 100 == 0:
            print print_timestamp() + "Finish %d scholar profile collecting" % cnt
            sys.stdout.flush()

        if cnt == 2:
            break

    # save files
    with open(SCHOLAR_PUBS, 'w') as fp:
        json.dump(scholar_profile, fp)


def collect_grants_abstract():
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
    with open(SCHOLAR_GRANTS, 'w') as fp:
        json.dump(scholar_award, fp)


if __name__ == "__main__":
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser(description="Description of data collecting")
    parser.add_argument("--type", type=str, help='Set the types of data for collecting: scholars publication(pub), '
                                                 'nsf grant abstract(nsf)', choices=['pub', 'nsf'])
    args = parser.parse_args()
    type = args.type

    if type == 'pub':
        collect_scholar_publications()
    elif type == 'nsf':
        collect_grants_abstract()
    else:
        parser.error('--type is not correct.')

