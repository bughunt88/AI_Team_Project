
# fetchall() : 지정 테이블 안의 모든 데이터를 추출
# fetchone() : 지정 테이블 안의 데이터를 한 행씩 추출
# fetchmany(size=원하는 데이터 수) : 지정 테이블 안의 데이터를 size 개의 행을 추출

import pymysql

connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()



    