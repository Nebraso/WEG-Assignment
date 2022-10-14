import socket
from flask import Flask, request, render_template
from model import fromUserToBackend

app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function action="{{ url_for("gfg")}}"


@app.route('/', methods=["GET", "POST"])
def gfg():

    if request.method == "POST":
        Gender = int(request.form.get("Gender"))
        Married = int(request.form.get("Married"))
        Dependents = int(request.form.get("Dependents"))
        if Dependents > 3:
            Dependents = 3
        Education = int(request.form.get("Education"))

        Employed_Self = int(request.form.get("Employed_Self"))
        ApplicantIncome = int(request.form.get("ApplicantIncome"))
        CoapplicantIncome = int(request.form.get("CoapplicantIncome"))
        LoanAmount = int(request.form.get("LoanAmount"))
        Loan_Amount_Term = int(request.form.get("Loan_Amount_Term"))
        History_Credit = int(request.form.get("History_Credit"))
       
        
        return fromUserToBackend(Gender, Married, Dependents, Education, Employed_Self,
                                 ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, History_Credit)
    return render_template("form.html")


if __name__ == '__main__':
    app.run()
