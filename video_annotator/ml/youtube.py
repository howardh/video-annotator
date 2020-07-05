import urllib.request
import json
import itertools

from config import config

api_key = config['youtube_v3_api_key']

base_url = "https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId=PLWBXNYRtvT0SZOppBJE0Hiai571MHFltz&key="+api_key

all_data = []

next_url = base_url
while next_url is not None:
    contents = urllib.request.urlopen(next_url).read()
    data = json.loads(contents)
    all_data.append(data)
    if 'nextPageToken' in data:
        next_url = base_url + '&pageToken=' + data['nextPageToken']
    else:
        next_url = None

# Get list of all videos and movements present in each
items = list(itertools.chain.from_iterable([d['items'] for d in all_data]))
data = []
for item in items:
    snippet = item['snippet']
    title = snippet['title']
    description = snippet['description'].lower()

    if len(title) != len('yyyy-mm-dd'):
        continue

    squat = 'squat' in description
    bench = 'bench' in description
    deadlift = 'deadlift' in description
    datum = (title, squat, bench, deadlift)
    data.append(datum)

# Check what's in the dataset
import re
import os
train_dataset_dir = 'dataset/videos/'
val_dataset_dir = 'smalldataset/videos/'
counts = [0,0,0]
for title,squat,bench,deadlift in data:
    m = re.search('(\d\d\d\d) (\d\d) (\d\d)', title)
    train_filename = os.path.join(train_dataset_dir, '%s-%s-%s.webm' % (m.group(1), m.group(2), m.group(3)))
    val_filename = os.path.join(val_dataset_dir, '%s-%s-%s.webm' % (m.group(1), m.group(2), m.group(3)))

    if os.path.isfile(train_filename):
        counts[0] += squat
        counts[1] += bench
        counts[2] += deadlift
        continue
    if os.path.isfile(val_filename):
        continue

    if squat or deadlift:
        print(title, squat, bench, deadlift)
print('Count of each movement in the dataset:')
print(counts)
