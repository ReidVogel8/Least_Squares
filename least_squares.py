#Question: Is there a linear or non-linear relationship between a student time spent studying and their exam score
#Link to dataset: http://kaggle.com/datasets/saadaliyaseen/analyzing-student-academic-trends


import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    #create x and y axis arrays
    #Hours studied
    x = []
    #Exam Score
    y = []

    #read in the file
    with open('student_exam_scores.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            hours_studied = float(row[1])  # hours studied
            score = float(row[5])  # score on exam
            x.append(hours_studied)
            y.append(score)


    #convert into numpy array
    x = np.array(x)
    y = np.array(y)

    #------linear fit code using cholesky decomposition--------
    #create Matrix A
    A = np.column_stack((np.ones_like(x), x))

    #form the normal equations
    ATA = A.T @ A
    ATy = A.T @ y

    #cholesky decomposition
    L = np.linalg.cholesky(ATA)

    #Solve L * y' = b(forward substitution)
    y_prime = np.linalg.solve(L, ATy)

    #Solve L.T * B = y'(backward substitution)
    beta = np.linalg.solve(L.T, y_prime)

    #getting the intercept and slope from beta which was solved in the back sub above
    intercept_chol, slope_chol = beta

    #------QR decomposition and solving for x using back substitution---------
    #back_substitution function
    def back_substitution(R, y):
        n = len(y)
        x = np.zeros_like(y)
        for i in reversed(range(n)):
            x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
        return x

    #qr decomposition
    Q, R = np.linalg.qr(A)
    y_qr = Q.T @ y

    beta_qr = back_substitution(R, y_qr)
    intercept_qr, slope_qr = beta_qr

    #-----LSQR check--------

    #this is the equivalent to matlab lsqr method.
    beta_lsqr, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    intercept_lsqr, slope_lsqr = beta_lsqr

    #-----Condition number-------
    condition_number = np.linalg.cond(A)
    print("Condition Number: {}".format(condition_number))
    #not a super large condition number but not considered well-conditioned

    #other curve fit
    # Fit a quadratic curve: Y = p1*x^2 + p2*x + p3
    p1, p2, p3 = np.polyfit(x, y, 2)

    #smoothig out the line
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = p1 * x_fit ** 2 + p2 * x_fit + p3

    #solving for the 3rd polynomial equation
    coefficients = np.polyfit(x, y, 3)
    third_fit = np.polyval(coefficients, x)


    # create graph and line of best fit
    plt.scatter(x, y, marker='o', color='blue', s=5)
    #plotting cholesky
    plt.plot(x, intercept_chol + slope_chol * x, color = 'green', linewidth = 3)
    #plotting qr in dotted red
    plt.plot(x, intercept_qr + slope_qr * x, color='red', linestyle='dotted', linewidth = 2)
    #plotting the lsqr method
    plt.plot(x, intercept_lsqr + slope_lsqr * x, color='orange', linestyle='dashdot', linewidth = 1)
    #Plotting the quadratic (poly2) fit
    plt.plot(x_fit, y_fit, color='magenta', linestyle='--', linewidth=2.5)
    #plotting cubic polynomial
    plt.plot(x, third_fit, color='purple', linestyle='-', linewidth=2)




    #hard to tell, but when both lines are plotted they are overlapping each other on the graph.
    #Both the chol and qr decompositions are giving similar answers

    #uncomment the two lines below to get the exact values above each point
    #for i, txt in enumerate(y):
        #plt.text(x[i], y[i], str(txt), ha='center', va='bottom')

    # Add labels and title
    plt.xlabel("Hours studied")
    plt.ylabel("Exam score")
    plt.title("Dot Plot showing the correlation between \n Hours studied and exam score")

    # Display the plot
    plt.show()

    #print out all the slopes and intercepts of the three methods
    print("----Cholesky Decomposition----\n Intercept Chol: {}".format(intercept_chol)+" slope: {}".format(slope_chol))
    print("----QR Decomposition----\n Intercept QR: {}".format(intercept_qr) + " slope: {}".format(slope_qr))
    print("----LSQR----\n Intercept lsqr: {}".format(intercept_lsqr) + " slope: {}".format(slope_lsqr))