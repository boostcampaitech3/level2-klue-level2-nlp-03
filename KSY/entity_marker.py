import pandas as pd


# added by sujeong;
def add_special_token(tokenizer, marker_type):
    print('Add Special Token...')
    # marker_types = ["entity_marker", "entity_marker_punc", "typed_entity_marker", "typed_entity_marker_punc"]
    # marker_type = marker_types[marker_type_num-1]

    if marker_type == "entity_marker":  # --> [E1] Bill [/E1] was born in [E2] Seattle [/E2].
        markers = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": markers})

    elif marker_type == "entity_marker_punc":  # --> @ Bill @ was born in # Seattle #.
        markers = ["@", "#"]
        added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": markers})

    elif marker_type == "typed_entity_marker":  # --> <S:PERSON> Bill </S:PERSON> was born in <O:CITY> Seattle </O:CITY>
        entity_types = ['PER', 'ORG', 'POH', 'DAT', 'LOC', 'NOH']
        markers = []
        for word_type in entity_types:
            sub_obj.append(f'<S:{word_type}>')
            sub_obj.append(f'</S:{word_type}>')
            sub_obj.append(f'<O:{word_type}>')
            sub_obj.append(f'</O:{word_type}>')
        added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": markers})

    elif marker_type == "typed_entity_marker_punc":  # --> @ +person+ Bill @ was born in #^city^ Seattle #.
        entity_types = ['PER', 'ORG', 'POH', 'DAT', 'LOC', 'NOH']
        markers = []
        for word_type in entity_types:
            sub_obj.append(f'@+{word_type}+')
            sub_obj.append(f'#^{word_type}^')
        sub_obj.append('@')
        sub_obj.append('#')
        added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": markers})
    print('Done!')
    return added_token_num, tokenizer


# added by sujeong;
def entity_marking(entity: str, is_subj: bool,
                   marker_type: str):  # entity는 Entity에 대한 dictionary 정보로 받는다. (type도 필요하기 때문)
    word, word_type = eval(entity)['word'], eval(entity)['type']

    marked_word = ''
    if marker_type == "entity_marker":  # --> [E1] Bill [/E1] was born in [E2] Seattle [/E2].
        if is_subj:
            marked_word = '[E1]' + word + '[/E1]'
        else:
            marked_word = '[E2]' + word + '[/E2]'

    elif marker_type == "entity_marker_punc":  # --> @ Bill @ was born in # Seattle #.
        if is_subj:
            marked_word = '@' + word + '@'
        else:
            marked_word = '#' + word + '#'

    elif marker_type == "typed_entity_marker":  # --> <S:PERSON> Bill </S:PERSON> was born in <O:CITY> Seattle </O:CITY>
        if is_subj:
            marked_word = f'<S:{word_type}>' + word + f'</S:{word_type}>'
        else:
            marked_word = f'<O:{word_type}>' + word + f'</O:{word_type}>'

    elif marker_type == "typed_entity_marker_punc":  # --> @ +person+ Bill @ was born in #^city^ Seattle #.
        if is_subj:
            marked_word = f'@+{word_type}+' + word + '@'
        else:
            marked_word = f'#^{word_type}^' + word + '#'
    return marked_word


# added by sujeong;
def get_entity_marked_data(df: pd.DataFrame, marker_type: str):
    print('Add Entity Marker to Data...')
    assert marker_type in ["entity_marker", "entity_marker_punc", "typed_entity_marker",
                           "typed_entity_marker_punc"], "marker type을 확인하세요."

    df_entity_marked = pd.DataFrame(columns=['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'])

    for i, data in enumerate(df.iloc):
        data_dict = dict(data)
        i_d, sentence, subj, obj, label, source = data_dict.values()
        subj_word, subj_start, subj_end, subj_type = eval(subj).values()
        obj_word, obj_start, obj_end, obj_type = eval(obj).values()

        # 1. Entity Marker 방식
        # marker_types을 바로 str으로 받는 것으로 변경
        # marker_types = ["entity_marker", "entity_marker_punc", "typed_entity_marker", "typed_entity_marker_punc"]
        # marker_type = marker_types[marker_type_num-1] # marker type num 바꾸면 이 부분이 바뀐다.

        # 2. Entity 찾기
        if subj_word != sentence[subj_start: subj_end + 1]:
            '''(똑같은 단어들 중 원래 idx와 가장 가까이 있는 애를 진짜 entity라고 생각)'''
            candidates = re.finditer(subj_word, sentence)
            closest_one = min(candidates, key=lambda x: abs(x.start() - subj_start) + abs(x.end() - (subj_end + 1)))
            subj_start, subj_end = closest_one.start(), closest_one.end() - 1

        # 3. Entity 찾기
        if obj_word != sentence[obj_start: obj_end + 1]:
            '''(똑같은 단어들 중 원래 idx와 가장 가까이 있는 애를 진짜 entity라고 생각)'''
            candidates = re.finditer(obj_word, sentence)
            closest_one = min(candidates, key=lambda x: abs(x.start() - obj_start) + abs(x.end() - (obj_end + 1)))
            obj_start, obj_end = closest_one.start(), closest_one.end() - 1

        # 4. Entity 표시 -> subj, obj 한 번에 해야 idx로 할 수 있음
        if subj_end < obj_start:
            sentence = sentence[:subj_start] \
                       + entity_marking(entity=subj, is_subj=True, marker_type=marker_type) \
                       + sentence[subj_end + 1:obj_start] \
                       + entity_marking(entity=obj, is_subj=False, marker_type=marker_type) \
                       + sentence[obj_end + 1:]
        elif obj_end < subj_start:
            sentence = sentence[:obj_start] \
                       + entity_marking(entity=obj, is_subj=False, marker_type=marker_type) \
                       + sentence[obj_end + 1:subj_start] \
                       + entity_marking(entity=subj, is_subj=True, marker_type=marker_type) \
                       + sentence[subj_end + 1:]

        # 5. 가공한 데이터 추가
        df_entity_marked.loc[i] = [i_d, sentence, subj, obj, label, source]
    print('Done!')
    return df_entity_marked