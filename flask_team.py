from flask import Flask, render_template , request
import numpy as np
import db_connect as db
import pandas as pd

def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    return dataset


app = Flask(__name__)
 
@app.route('/')

def home():
    return render_template("map.html")


@app.route('/view')

def view():
    name = request.args.get("name")
    
    #x_pred = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date ", is_train = False)
    #x_train, y_train = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

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

    location += "'"

    sql = "SELECT date,YEAR,MONTH,day,time,category,dong,value, IFNULL((SELECT VALUE FROM main_data_table AS s WHERE  s.date = DATE_SUB(a.date, INTERVAL 1 YEAR) AND a.month=s.month AND a.time=s.time AND a.category=s.category AND a.dong = s.dong ),0) AS l_value from result_data_table as a where 1"+location
    sql += " order by a.date, a.year, a.month, a.day, a.time asc"
    print(sql)
    x_pred = load_data(sql)
    print(x_pred)

    return render_template("table.html", subject = name, data = x_pred)
 
if __name__=='__main__':
 app.run(host='0.0.0.0', port=5000, debug=True)
