from flask import Flask, render_template , request
import numpy as np
import pandas as pd
import pymysql

def load_data(query, is_train = True):

    connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
    cur = connect.cursor()
    query = query
    cur.execute(query)
    dataset = np.array(cur.fetchall())
    cur.close()

    return dataset


app = Flask(__name__)
 
@app.route('/')

def home():
    return render_template("map.html")


@app.route('/view')

def view():
    name = request.args.get("name")
    cate = str(request.args.get("cate"))

    x_pred = []

    if cate != "None":

        location = " and a.dong ='"
        
        if name =='구로구':
            location += "0"
        elif name =='영등포구':
            location += "1"
        elif name =='도봉구':
            location += "2"
        elif name =='성북구':
            location += "3"
        elif name =='동작구':
            location += "4"
        elif name =='서대문구':
            location += "5"
        elif name =='노원구':
            location += "6"
        elif name =='은평구':
            location += "7"
        elif name =='마포구':
            location += "8"
        elif name =='금천구':
            location += "9"
        elif name =='강서구':
            location += "10"
        elif name =='양천구':
            location += "11"
        elif name =='강북구':
            location += "12"
        elif name =='관악구':
            location += "13"
        elif name =='송파구':
            location += "14"
        elif name =='강남구':
            location += "15"
        elif name =='서초구':
            location += "16"
        elif name =='용산구':
            location += "17"
        elif name =='동대문구':
            location += "18"
        elif name =='강동구':
            location += "19"
        elif name =='중랑구':
            location += "20"
        elif name =='중구':
            location += "21"
        else:
            location += "22"

        location += "'"

        cate_list = ""

        if cate != "None":
            cate_list += " and a.category = '" + cate + "' "


        sql = "SELECT DATE_ADD(a.DATE, INTERVAL 1 MONTH) AS m_date, a.YEAR,a.MONTH,a.DAY,a.TIME,a.category,a.dong,a.VALUE, IFNULL((SELECT VALUE FROM main_data_table AS s WHERE  s.date = DATE_ADD(DATE_SUB(a.date, INTERVAL 1 YEAR),INTERVAL 2 DAY) AND a.month=s.month AND a.time=s.time AND a.category=s.category AND a.dong = s.dong ),0) AS l_value FROM result_data_table AS a LEFT JOIN main_data_table AS s ON s.date = DATE_SUB(a.date, INTERVAL 1 YEAR) AND a.month=s.month AND a.time=s.time AND a.category=s.category AND a.dong = s.dong  where 1"+location + cate_list
        sql += " order by a.date, a.year, a.month, a.day, a.time asc"
        x_pred = load_data(sql)

    return render_template("table.html", subject = name, data = x_pred, cate = cate)
 
if __name__=='__main__':
 app.run(host='0.0.0.0', port=5000, debug=True)
 #app.run()

