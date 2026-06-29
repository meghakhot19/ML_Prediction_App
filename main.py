from flask import Flask, render_template,request
import numpy as np
import pickle

app = Flask(__name__)

with open("diabetes.pkl", "rb") as file:
    diabetes_model = pickle.load(file)

with open("heart.pkl", "rb") as file:
    heart_model = pickle.load(file)

with open("tennis.pkl", "rb") as file:
    tennis_model = pickle.load(file)

with open("titanic.pkl", "rb") as file:
    titanic_model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diabetes", methods=["GET", "POST"])
def model1():

    if request.method == "GET":
        return render_template("diabetes.html")

    try:
        a = float(request.form["n1"])
        b = float(request.form["n2"])
        c = float(request.form["n3"])
        d = float(request.form["n4"])
        e = float(request.form["n5"])
        f = float(request.form["n6"])
        g = float(request.form["n7"])
        h = float(request.form["n8"])

        data = np.array([[a,b,c,d,e,f,g,h]])

        prediction = diabetes_model.predict(data)

        result = "Diagnose" if prediction[0] == 1 else "Not Diagnose"

        return render_template(
            "diabetes.html",
            prediction_text=result
        )

    except Exception as e:
        return render_template(
            "diabetes.html",
            prediction_text=f"Error: {e}"
        )
    
@app.route("/heart", methods=["GET", "POST"])
def model2():

    if request.method == "GET":
        return render_template("heart.html")

    try:
       a = float(request.form["age"])
       b = float(request.form["sex"])
       c = float(request.form["cp"])
       d = float(request.form["trestbps"])
       e = float(request.form["chol"])
       f = float(request.form["fbs"])
       g = float(request.form["restecg"])
       h = float(request.form["thalach"])
       i = float(request.form["exang"])
       j = float(request.form["oldpeak"])
       k = float(request.form["slope"])
       l = float(request.form["ca"])
       m = float(request.form["thal"])

       data = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m]])

       prediction = heart_model.predict(data)

       result = "Diagnose" if prediction[0] == 1 else "Not Diagnose"

       return render_template(
            "heart.html",
            prediction_text=result
        )

    except Exception as e:
        return render_template(
            "heart.html",
            prediction_text=f"Error: {e}"
        )
    
@app.route("/tennis", methods=["GET", "POST"])
def tennis():

    if request.method == "GET":
        return render_template("tennis.html")

    try:
        a = float(request.form["day"])
        b = float(request.form["outlook"])
        c = float(request.form["temp"])
        d = float(request.form["humidity"])
        e = float(request.form["wind"])

        data = np.array([[a, b, c, d, e]])

        prediction = tennis_model.predict(data)

        result = "Play Tennis" if prediction[0] == 1 else "Don't Play Tennis"

        return render_template(
            "tennis.html",
            prediction_text=result
        )

    except Exception as e:
        return render_template(
            "tennis.html",
            prediction_text=f"Error: {e}"
        )
    
@app.route("/titanic", methods=["GET", "POST"])
def model4():

    if request.method == "GET":
        return render_template("titanic.html")

    try:
        a = int(request.form["pclass"])
        b = float(request.form["sex"])
        c = float(request.form["age"])
        d = int(request.form["sibsp"])
        e = float(request.form["parch"])
        f = float(request.form["fare"])
        g = float(request.form["embarked"])

        data = np.array([[a, b, c, d, e, f, g]])

        prediction = titanic_model.predict(data)

        if prediction[0] == 1:
            result = " Passenger Survived"
        else:
            result = " Passenger Did Not Survive"

        return render_template(
            "titanic.html",
            prediction_text=result
        )

    except Exception as e:
        return render_template(
            "titanic.html",
            prediction_text=f"Error: {e}"
        )

if __name__=="__main__":
    app.run(debug=True)