# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
UNIQUE_GENRE_FOLDING = {'Religion': ['Religious Fiction', 'Christianity - Biographies', 'Christian Fiction', 'Weymouth New Testament', 'Other religions', 'World English Bible', 'Christianity - Commentary', 'Atheism & Agnosticism', 'Bibles', 'Religion', 'Christianity - Other'],
                        'Poetry': ['Poetry', 'Ballads', 'Antiquity', 'Lyric', 'Anthologies', 'Multi-version (Weekly and Fortnightly poetry)', 'Elegies & Odes', 'Free Verse', 'Sonnets'],
                        'Theater': ['Tragedy', 'Plays', 'Comedy', 'Performing Arts'],
                        'Ancient': ['Classics (Greek & Latin Antiquity)', 'Medieval', 'Ancient'],
                        'Fiction': ['General Fiction', 'Gothic Fiction', 'Sagas', 'Fantasy Fiction', 'Legends & Fairy Tales', 'Suspense', 'Espionage', 'Political & Thrillers', 'Myths', 'Fantastic Fiction', 'Fictional Biographies & Memoirs', 'Literary Collections', 'True Crime', 'Short Stories', 'Epics', 'Culture & Heritage Fiction', 'Nature & Animal Fiction', 'War & Military Fiction', 'Sports Fiction', 'Romance', 'Action & Adventure',  "Children's Fiction", 'Erotica', 'Drama', 'Historical Fiction', 'Westerns', 'Detective Fiction', 'Literary Fiction', 'Horror & Supernatural Fiction', 'Crime & Mystery Fiction', 'Science Fiction', 'Narratives', 'Travel Fiction', 'Epistolary Fiction', 'Action & Adventure Fiction', 'Nautical & Marine Fiction'],
                        'Non fiction': ['Biography & Autobiography', 'Middle Ages/Middle History', 'Letters', 'Historical', 'Exploration', "Children's Non-fiction", 'Short non-fiction', 'History', 'Memoirs', 'Non fiction'],
                        'Humor': ['Humor', 'Satire', 'Humorous Fiction'],
                        'Essay': ['Travel & Geography', 'Essays & Short Works', 'Law', 'Family Life', 'Psychology', 'School', 'War & Military', 'Transportation', 'Games', 'Art', 'Family & Relationships', 'Essays', 'Business & Economics', 'Education', 'Arts', 'Animals & Nature', 'Philosophy', 'Literary Criticism', 'Writing & Linguistics'],
                        'Craft': ['Cooking', 'Crafts & Hobbies', 'Self-Help', 'Music', 'Gardening', 'Sports & Recreation', 'House & Home', 'Health & Fitness', 'Design & Architecture'],
                        'Dramatic Readings': ['Dramatic Readings'],
                        'Science': ['Mathematics', 'Science', 'Medical', 'Astronomy', 'Earth Sciences', 'Physics & Mechanics', 'Political Science', 'Life Sciences', 'Chemistry', 'Language learning', 'Social Science (Culture & Anthropology)', 'Technology & Engineering', 'Nature', 'Animals'],
                        'Undefined': ['Modern (20th C)', 'Reference', 'Douay-Rheims Version', 'Short works', 'Modern', 'Single Author Collections', 'null', 'General', 'Early Modern', 'Modern (19th C)', 'Published 1800 -1900', '*Non-fiction', 'Contemporary', 'Family', 'Single author',  'Published before 1800', 'Published 1900 onward', "Young's Literal Translation", 'King James Version']}

GENDER_ORDERING = ['Dramatic Readings', 'Poetry', 'Religion', 'Theater', 'Fiction',
                   'Science', 'Essay', 'Humor', 'Ancient', 'Non fiction', 'Craft', 'Undefined']

SUPER_GENDER_FOLDING = {'Science, Craft & Essay': ['Essay', 'Science', 'Craft'],
                        'Literature': ['Fiction', 'Non fiction', 'Humor']}

SUPER_GENDER_ORDERING = ['Dramatic Readings', 'Poetry', 'Religion',
                         'Literature', 'Science, Craft & Essay', 'Theater', 'Ancient', 'Undefined']
