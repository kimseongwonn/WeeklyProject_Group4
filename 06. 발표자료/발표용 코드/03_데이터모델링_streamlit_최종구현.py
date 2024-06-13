import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler
import random
from random import shuffle

# 전체 데이터 로드
all_data = pd.read_csv('./data/final_data_set_v5_링크수정.csv', encoding='cp949')

all_data.rename({'Longitude': 'longitude', 'Latitude': 'latitude'}, axis=1, inplace=True)
lda = pd.read_csv('./data/LDA_tour_DB.csv', index_col=0)

food_ratio = pd.read_csv('./data/food_ratio_result.csv', index_col=0)
tour_ratio = pd.read_csv('./data/tour_ratio_result.csv', index_col=0)

# 초기 상태 설정
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'selected_attraction' not in st.session_state:
    st.session_state.selected_attraction = None

# 함수 부분 설정


def map_hashtags(hashtags, category):
    tour_hashtag_mapping = {
        '#분위기': '매력도', '#시설': '편의', '#활동': '매력도',
        '#접근성': '편의', '#서비스': '만족도', '#가격': '만족도'
    }
    food_hashtag_mapping = {
        '#음식의 맛과 질': '음식의 속성', '#음식의 다양성': '음식의 속성',
        '#시설 및 환경': '시설 및 분위기', '#분위기': '시설 및 분위기',
        '#청결도와 위생': '시설 및 분위기', '#접근성': '시설 및 분위기',
        '#특별한 목적': '시설 및 분위기', '#가격과 가치': '가격 및 서비스',
        '#서비스': '가격 및 서비스'
    }

    if category == '관광':
        hashtag_mapping = tour_hashtag_mapping
    else:
        hashtag_mapping = food_hashtag_mapping

    mapped_hashtags = set(hashtag_mapping.get(hashtag, '')
                          for hashtag in hashtags if hashtag in hashtag_mapping)

    return list(mapped_hashtags)  # 결과를 다시 리스트로 반환하여 순서가 있는 데이터 구조 제공


def making_pivot(category, popularity_class, types, hashtags, all_data):
    mapped_hashtags = map_hashtags(hashtags, category)

    # 데이터 필터링
    filtered_data = all_data[(all_data["Category"] == category) &
                             (all_data["class"] == popularity_class) &
                             (all_data["type"].isin(types)) &
                             (all_data["Category_Map"].isin(mapped_hashtags))]

    if not filtered_data.empty:
        pivot_table = filtered_data.pivot_table(
            index='Attraction', columns='Category_Map', values='Selected_People', aggfunc='sum', fill_value=0)

    if category == '관광':
        pivot_table = pd.merge(pivot_table, tour_ratio)
    else:
        pivot_table = pd.merge(pivot_table, food_ratio)
        return pivot_table.drop_duplicates()


def making_pivot_min(df, category):
    if category == '관광':
        df = df[df['Category'] == '관광']
    elif category == '음식':
        df = df[df['Category'] == '음식']

    return df.pivot_table(index='Attraction', columns='Category_Map', values='Selected_People', aggfunc='sum', fill_value=0).reset_index()

# 유사도 함수


def cosine_similarity_func(df, df2, category, types, attraction, top_n=5):

    condition = (df['Category'] == category) & (df['type'].isin(types))
    df_con = df[condition]
    temp = making_pivot_min(df_con, category)
    if category == '관광':
        feature_data = pd.merge(temp, df2).iloc[:, 7:10].values
    else:
        feature_data = pd.merge(temp, df2).iloc[:, 10:13].values
    cosine_sim = cosine_similarity(feature_data)
    index = temp[temp['Attraction'] == attraction].index[0]
    cosine_sim[index, index] = -np.inf
    most_similar_indices = np.argsort(cosine_sim[index])[-top_n:]
    recommendations = temp.iloc[most_similar_indices]['Attraction'].values.tolist(
    )
    return recommendations

# 랜덤 테마여행


def random_tour(df, topic):
    random.seed(42)
    tp1 = df[df['topic_type'] == topic]
    tp1_lst = tp1[tp1['max'] >= 0.95]['Attraction'].values
    shuffle(tp1_lst)
    tp1_lst = tp1_lst[:5].tolist()
    return tp1_lst


# css 설정
# 사진 후보 1 : https://i.imgur.com/tKIstK1.jpeg
# 사진 후보 2 : https://i.imgur.com/T4tLYtH.jpeg

# def add_bg_from_url():
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("https://i.imgur.com/T4tLYtH.jpeg");
#             background-attachment: fixed;
#             background-size: 100% auto;  # 가로 길이를 화면 너비에 맞춤
#             background-position: center center;  # 이미지를 세로 및 가로 중앙에 위치
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )


# add_bg_from_url()


#########################  1페이지 ###################################
def page1():
    st.write(' ')
    st.write('')
    st.title('여러분의 취향에 맞는 여행지를 골라드립니다')
    st.write('# by E1i4 # 서울여행 # 취향저격 # 관광지부터 # 맛집까지')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    if st.button(' :grey_exclamation: 가보자고 :grey_exclamation: 봐보자고 :grey_exclamation: 떠나보자고 :grey_exclamation: 즐겨보자고 :grey_exclamation: '):
        st.session_state.page += 1
        st.experimental_rerun()

    #########################  2페이지 ###################################
    # 관광/음식 : category


def page2():
    st.title('당신은 어떤 여행가이신가요?')
    st.slider('여행지 검색중... :female-technologist: ',
              min_value=1, max_value=7, value=1)
    con2, con3 = st.columns([5, 5])
    st.write(' ')
    st.write(' ')
    st.write(' ')
    with con2:
        st.subheader('다양한 장소에 가고 싶어!')
        st.write('- 탐험가 유형 ')
        if st.button(" 다양한 관광지 보러가기 :mountain_railway: "):
            st.session_state.category = '관광'
            st.session_state.page += 1
            st.experimental_rerun()
    with con3:
        st.subheader('금강산도 식후경이지!')
        st.write('- 미식가 유형 ')
        if st.button(" 취향저격 맛집 보러가기 :hamburger:"):
            st.session_state.category = '음식'
            st.session_state.page += 1
            st.experimental_rerun()


#########################  2페이지 ###################################
# 인기도 : popularity_class

def page3():
    st.title("'핫 :fire: 플레이스' 부터 '숨은 명소' :mag: ")
    st.slider('같이갈 친구 구하는중... :women_holding_hands: ',
              min_value=1, max_value=7, value=2)
    st.empty()
    st.write(' ')
    st.write(' ')
    st.write(' ')
    con1, con2, con3 = st.columns([3, 3, 3])
    with con1:
        st.write('- 많이 가는 곳에는 이유가 있는법')
        if st.button('핫플레이스 보러가기 :fire:'):
            st.session_state.popularity_class = 1
            st.session_state.page += 1
            st.experimental_rerun()
    with con2:
        st.write('- 뭐든 적당한게 좋아')
        if st.button('대중적인 관광지 보러가기 :passenger_ship:'):
            st.session_state.popularity_class = 2
            st.session_state.page += 1
            st.experimental_rerun()
    with con3:
        st.write('- 숨겨진 관광지라고 무시하면 안된다구')
        if st.button('나만 알고 싶은 관광지 보러가기 :grinning_face_with_star_eyes:'):
            st.session_state.popularity_class = 3
            st.session_state.page += 1
            st.experimental_rerun()
    if st.button("처음으로"):
        st.session_state.page = 0
        st.experimental_rerun()

#########################  3페이지 ###################################
# 카테고리 선택  : types


def page4():
    st.title("어떤 타입의 여행을 원하시나요?")
    st.slider('티켓 예매하는 중... :admission_tickets: ',
              min_value=1, max_value=7, value=3)
    category = st.session_state.category
    st.write(' ')
    st.write(' ')
    st.write(' ')
    if category == '관광':
        types = st.multiselect("가장 원하는 여행 타입을 골라주세요",

                               ['experience', 'nature', 'theme',
                                   'history', 'shop_etc', 'culture'])
    else:
        types = st.multiselect("가장 원하는 여행 타입을 골라주세요",

                               ['cafe', 'korean', 'chinese',
                                   'west', 'japanese', 'etc'])

    con1, con2, con3 = st.columns([3, 3, 3])
    with con1:
        if st.button("처음으로"):
            st.session_state.page = 0
            st.experimental_rerun()
    with con2:
        if st.button("이전"):
            st.session_state.page -= 1
            st.experimental_rerun()
    with con3:
        if st.button("다음"):
            if not types:
                st.warning("해시태그를 하나 이상 선택하세요.")
            else:
                st.session_state.types = types
                st.session_state.page += 1
                st.experimental_rerun()


############################  5페이지 ###################################
# 해쉬테그 선택 : hashtags

def page5():
    st.title("이것만큼은 꼭 :heavy_exclamation_mark: 고려되었으면 하는 여행지 속성이 있으신가요?")
    st.slider('자동차 시동거는 중..	 :car: ', min_value=1, max_value=7, value=4)
    category = st.session_state.category
    st.write(' ')
    st.write(' ')
    st.write(' ')
    if category == '관광':
        hashtags = st.multiselect('여행지 선택시 가장 중요하게 고려하는 부분을 선택해주세요',

                                  ['#분위기', '#시설', '#활동', '#접근성', '#서비스', '#가격'])

    else:
        hashtags = st.multiselect('여행지 선택시 가장 중요하게 고려하는 부분을 선택해주세요',

                                  ['#음식의 맛과 질', '#음식의 다양성', '#시설 및 환경', '#분위기',
                                   '#청결도와 위생', '#접근성', '#특별한 목적', '#가격과 가치', '#서비스'])
    con1, con2, con3 = st.columns([3, 3, 3])
    with con1:
        if st.button("처음으로"):
            st.session_state.page = 0
            st.experimental_rerun()
    with con2:
        if st.button("이전"):
            st.session_state.page -= 1
            st.experimental_rerun()
    with con3:
        if st.button("다음"):
            if not hashtags:
                st.warning("해시태그를 하나 이상 선택하세요.")
            else:
                st.session_state.hashtags = hashtags
                st.session_state.page += 1
                st.experimental_rerun()

############################  5페이지 ###################################
# pivot_table : 필터된 테이블
# 여기부터 수정
# image_link /  hash_lst /


def page6():
    st.title("당신의 취향을 반영한 여행지를 추천합니다")
    st.slider('친구랑 한잔 하는 중... :beer: ', min_value=1, max_value=7, value=5)
    category = st.session_state.get('category', '관광')
    popularity_class = st.session_state.get('popularity_class', 0)
    types = st.session_state.get('types', [])
    hashtags = st.session_state.get('hashtags', [])
    # 일단 피봇됨
    pivot_table = making_pivot(
        category, popularity_class, types, hashtags, all_data)
    hashtags = map_hashtags(hashtags, category)

    hash = list()
    for i in hashtags:
        i = i + ' 비율'
        hash.append(i)

    if not pivot_table.empty:
        st.write("어떤 기준으로 보고 싶으세요? 기준별 인기순으로 나열합니다")
        columns = st.columns(len(hash))

        for i, hash_column in enumerate(hash):
            if columns[i].button(hash_column, key=f'button_{i}_{hash_column}'):
                sorted_pivot_table = pivot_table.sort_values(
                    by=hash_column, ascending=False)
                st.session_state['sorted_pivot_table'] = sorted_pivot_table
                st.experimental_rerun()

        display_table = st.session_state.get('sorted_pivot_table', pivot_table)

        for idx, row in display_table.iterrows():
            attraction = row['Attraction']
            cols = st.columns([5, 5])
            cols[0].write(attraction)

            # 이미지 링크, 해시태그 및 정보 가져오기
            image_link = all_data.loc[all_data['Attraction']
                                      == attraction, 'Image Link'].values
            hash_lst = all_data.loc[all_data['Attraction']
                                    == attraction, 'Tag'].values
            info_lst = all_data.loc[all_data['Attraction']
                                    == attraction, 'info'].values

            with cols[0]:
                if len(image_link) > 0:
                    st.image(image_link[0], caption=attraction,
                             use_column_width=True, output_format='auto')

            with cols[1]:
                if st.button(attraction, help=info_lst[0] if len(info_lst) > 0 else ""):
                    st.session_state.selected_attraction = attraction
                    st.session_state.page += 1
                    st.experimental_rerun()
                if len(hash_lst) > 0:
                    st.write(hash_lst[0])

    con1, con3 = st.columns([5, 5])
    with con1:
        if st.button("이전"):
            st.session_state.page -= 1
            st.experimental_rerun()
    with con3:
        if st.button("처음으로"):
            st.session_state.page = 0
            st.experimental_rerun()


############################  6페이지 ###################################
# 장소선택


def page7():
    attraction = st.session_state.selected_attraction
    st.title(attraction)
    st.slider('숙취로 고생하는 중... :face_vomiting: ',
              min_value=1, max_value=7, value=6)
    types = st.session_state.types
    if attraction:

        con7, con8 = st.columns([5, 5])
        with con7:
            image_link = all_data.loc[all_data['Attraction']
                                      == attraction, 'Image Link'].values
            st.image(image_link[0], caption=attraction,
                     use_column_width=True, output_format='auto')

        with con8:
            # 상세 주소
            address_info = all_data.loc[all_data['Attraction']
                                        == attraction, 'Address'].values[0]
            st.write(f'상세 주소 : {address_info}')
            # 관광지 소개
            info_sentence = all_data.loc[all_data['Attraction']
                                         == attraction, 'info'].values[0]
            st.write(info_sentence)

        # 유사도 추천
        st.subheader('당신이 선택한 속성들을 고려하여 선정된 유사한 장소입니다')
        category = st.session_state.category
        if category == '관광':
            cos_attraction = cosine_similarity_func(
                all_data, tour_ratio, category, types, attraction)
        else:
            cos_attraction = cosine_similarity_func(
                all_data, food_ratio, category, types, attraction)
        con11, con22, con33, con44, con55 = st.columns([2, 2, 2, 2, 2])
        with con11:
            if st.button(cos_attraction[0]):
                st.session_state.selected_attraction = cos_attraction[0]
                st.experimental_rerun()
        with con22:
            if st.button(cos_attraction[1]):
                st.session_state.selected_attraction = cos_attraction[1]
                st.experimental_rerun()
        with con33:
            if st.button(cos_attraction[2]):
                st.session_state.selected_attraction = cos_attraction[2]
                st.experimental_rerun()

        with con44:
            if st.button(cos_attraction[3]):
                st.session_state.selected_attraction = cos_attraction[3]
                st.experimental_rerun()
        with con55:
            if st.button(cos_attraction[4]):
                st.session_state.selected_attraction = cos_attraction[4]
                st.experimental_rerun()

        st.subheader(f'{attraction} 의 근처에는 이런 장소도 추천해요 :thumbsup: ')
        # 주어진 attraction의 주소에서 동 이름 추출
        dong = ' '.join(all_data.loc[all_data['Attraction']
                                     == attraction, 'Address'].values[0].split(' ')[:2])
        filtered_df = all_data[all_data['Address'].str.contains(dong)]

        # 폴리움 맵 생성
        center_lat = all_data.loc[all_data['Attraction']
                                  == attraction, 'latitude'].values[0]
        center_lon = all_data.loc[all_data['Attraction']
                                  == attraction, 'longitude'].values[0]

        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

        main_location = all_data[all_data['Attraction'] ==
                                 attraction][['latitude', 'longitude']].iloc[0]
        folium.Marker(location=[main_location['latitude'], main_location['longitude']],
                      popup=attraction, tooltip=attraction, icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

        for idx, row in filtered_df.iterrows():
            if row['Attraction'] != attraction:
                folium.Marker(location=[row['latitude'], row['longitude']],
                              popup=row['Address'], tooltip=row['Attraction']).add_to(m)

        folium_static(m)

    columns_page = st.columns(3)
    with columns_page[0]:
        if st.button('이전'):
            st.session_state.page -= 1
            st.experimental_rerun()
    with columns_page[1]:
        if st.button('처음으로'):
            st.session_state.page = 0
            st.experimental_rerun()
    with columns_page[2]:
        if st.button('테마 추천 받기'):
            st.session_state.page += 1
            st.experimental_rerun()

############################  7페이지 ###################################


def page8():
    st.subheader('혹시! 이건 어떠세요?')
    st.title('E1i4 가 추천하는 테마 여행지 :ferris_wheel: ')
    st.slider('여행후 집에서 꿀잠중.. :bed: ', min_value=1, max_value=7, value=7)
    st.write('#가족 #아이 #체험 #나들이')
    st.subheader('아이와 함께하는 서울 나들이 :baby: ')
    zero_lst = random_tour(lda, 'Topic 0')  # 주제에 맞는 명소 리스트 생성
    columns_zero = st.columns(5)  # 5개의 컬럼 생성

    # 각 명소에 대해 이미지와 버튼을 배치
    for i, attraction in enumerate(zero_lst):
        with columns_zero[i]:
            image_link = all_data.loc[all_data['Attraction']
                                      == attraction, 'Image Link'].values
            if image_link.any():
                st.image(image_link[0], caption=attraction,
                         use_column_width=True, output_format='auto')
            if st.button(attraction, key=f'btn_topic0_{i}'):
                st.session_state.selected_attraction = attraction
                st.session_state.page += 1
                st.experimental_rerun()

    st.empty()
    st.write('#여행 #문화 #역사 #친구 #관광지')
    st.subheader('함께하는 역사와 문화 여행 :classical_building: ')
    zero_lst = random_tour(lda, 'Topic 1')
    columns_one = st.columns(5)
    for i, attraction in enumerate(zero_lst):
        with columns_one[i]:
            image_link = all_data.loc[all_data['Attraction']
                                      == attraction, 'Image Link'].values
            if image_link.any():
                st.image(image_link[0], caption=attraction,
                         use_column_width=True, output_format='auto')
            if st.button(attraction, key=f'btn_topic1_{i}'):
                st.session_state.selected_attraction = attraction
                st.session_state.page += 1
                st.experimental_rerun()

    st.empty()
    st.write('#여행 #휴식 #공간 #공원 #산책 #도심')
    st.subheader('도심 속 힐링 공간 :herb: ')
    zero_lst = random_tour(lda, 'Topic 2')
    columns_two = st.columns(5)
    for i, attraction in enumerate(zero_lst):
        with columns_two[i]:
            image_link = all_data.loc[all_data['Attraction']
                                      == attraction, 'Image Link'].values
            if image_link.any():
                st.image(image_link[0], caption=attraction,
                         use_column_width=True, output_format='auto')
            if st.button(attraction, key=f'btn_topic2_{i}'):
                st.session_state.selected_attraction = attraction
                st.session_state.page += 1
                st.experimental_rerun()

    columns_three = st.columns(2)
    with columns_three[0]:
        if st.button("처음으로"):
            st.session_state.page = 0
            st.experimental_rerun()
    with columns_three[1]:
        if st.button('이전'):
            st.session_state.page -= 1
            st.experimental_rerun()


############################  8페이지 ###################################

def page9():
    attraction = st.session_state.selected_attraction
    st.title(attraction)
    st.slider('숙취로 고생하는 중... :face_vomiting: ',
              min_value=1, max_value=7, value=6)
    types = st.session_state.types
    if attraction:

        con7, con8 = st.columns([5, 5])
        with con7:
            image_link = all_data.loc[all_data['Attraction']
                                      == attraction, 'Image Link'].values
            st.image(image_link[0], caption=attraction,
                     use_column_width=True, output_format='auto')

        with con8:
            # 상세 주소
            address_info = all_data.loc[all_data['Attraction']
                                        == attraction, 'Address'].values[0]
            st.write(f'상세 주소 : {address_info}')
            # 관광지 소개
            info_sentence = all_data.loc[all_data['Attraction']
                                         == attraction, 'info'].values[0]
            st.write(info_sentence)

        st.subheader(f'{attraction} 의 근처에는 이런 장소도 추천해요 :thumbsup: ')
        # 주어진 attraction의 주소에서 동 이름 추출
        dong = ' '.join(all_data.loc[all_data['Attraction']
                                     == attraction, 'Address'].values[0].split(' ')[:2])
        filtered_df = all_data[all_data['Address'].str.contains(dong)]

        # 폴리움 맵 생성
        center_lat = all_data.loc[all_data['Attraction']
                                  == attraction, 'latitude'].values[0]
        center_lon = all_data.loc[all_data['Attraction']
                                  == attraction, 'longitude'].values[0]

        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

        main_location = all_data[all_data['Attraction'] ==
                                 attraction][['latitude', 'longitude']].iloc[0]
        folium.Marker(location=[main_location['latitude'], main_location['longitude']],
                      popup=attraction, tooltip=attraction, icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

        for idx, row in filtered_df.iterrows():
            if row['Attraction'] != attraction:
                folium.Marker(location=[row['latitude'], row['longitude']],
                              popup=row['Address'], tooltip=row['Attraction']).add_to(m)

        folium_static(m)

    con1, con2, con3 = st.columns([3, 3, 3])
    with con1:
        if st.button('테마 추천 받기'):
            st.session_state.page -= 1
            st.experimental_rerun()
    with con2:
        if st.button('처음으로'):
            st.session_state.page = 0
            st.experimental_rerun()
    with con3:
        if st.button('결정완료!'):
            st.session_state.page += 1
            st.experimental_rerun()


def page10():
    st.write()
    st.write()
    st.write()
    st.title('즐거운 여행 되시길 바랍니다')
    st.header('이상 E1i4 였습니다 :blush: 감사합니다!')
    st.image('/Users/leegahee/git_folder/04_프로젝트/2.주간프로젝트/마지막/진짜마지막/안녕짤.jpg')


if __name__ == "__main__":
    if st.session_state.page == 0:
        page1()
    elif st.session_state.page == 1:
        page2()
    elif st.session_state.page == 2:
        page3()
    elif st.session_state.page == 3:
        page4()
    elif st.session_state.page == 4:
        page5()
    elif st.session_state.page == 5:
        page6()
    elif st.session_state.page == 6:
        page7()
    elif st.session_state.page == 7:
        page8()
    elif st.session_state.page == 8:
        page9()
    elif st.session_state.page == 9:
        page10()
