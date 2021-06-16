from flask import Flask, render_template , request
app = Flask(__name__)
 
@app.route('/')

def home():
    return render_template("map.html")


@app.route('/view')

def view():
    name = request.args.get("name")

    print(name)

    return render_template("table.html", subject = name)
 
if __name__ == '__main__':
    
    app.run()
