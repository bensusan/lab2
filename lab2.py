import numpy as np
# import pandas
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline


def find_between( s, first, last ):
    start = s.index( first ) + len( first )
    end = s.index( last, start )
    return s[start:end]


def isNotEmpty(s):
    return bool(s and s.strip())


def predictweek(currenttransaction,alltransactions):
    if currenttransaction[9] < 8:
        max = 7
        min = 1
    elif currenttransaction[9] < 15:
        max = 14
        min = 8
    elif currenttransaction[9] < 22:
        max = 21
        min = 15
    else:
        max = 31
        min = 22
    maskuser = (alltransactions[:, 3] == currenttransaction[3])  # user id
    alltransactions = alltransactions[maskuser]
    maskamount = (alltransactions[:, 5].astype(float) > 0)  # amount
    alltransactions = alltransactions[maskamount]
    maskyear =  (alltransactions[:, 7] == currenttransaction[7])  # year
    maskmonth = (alltransactions[:, 8] == currenttransaction[8])  # month
    maskmaxday = (alltransactions[:,9].astype(float)<=max)  # day
    maskminday = (alltransactions[:,9].astype(float)>=min)  # day
    mask=np.logical_and(np.logical_and(np.logical_and(maskyear,maskmonth),maskmaxday),maskminday)
    alltransactions = alltransactions[mask]

    return np.sum(alltransactions[:,5].astype(float))

def predictmonth(currenttransaction, alltransactions):
    maskuser = (alltransactions[:, 3] == currenttransaction[3])  # user
    alltransactions=alltransactions[maskuser]
    maskamount = (alltransactions[:, 5].astype(float) > 0)  # amount
    alltransactions = alltransactions[maskamount]

    maskyear = (alltransactions[:, 7] == currenttransaction[7])
    maskmonth = (alltransactions[:, 8] == currenttransaction[8])

    alltransactions=alltransactions[np.logical_and(maskyear,maskmonth)]
    return np.sum(alltransactions[:,5].astype(float))

def predictsubscription(currenttransaction, alltransactions):
    maskuser = (alltransactions[:, 3] == currenttransaction[3])  # user
    alltransactions = alltransactions[maskuser]
    maxamount = currenttransaction[5].astype(float) * 1.1
    minamount = currenttransaction[5].astype(float) * 0.9
    maskmaxamount = (alltransactions[:, 5].astype(float) <= maxamount)  # amount
    maskminamount = (alltransactions[:, 5].astype(float) >= minamount)  # amount
    mask = np.logical_and(maskmaxamount, maskminamount)
    alltransactions = alltransactions[mask]
    for transaction1 in alltransactions:
        if transaction1[2]==currenttransaction[2]:
            continue
        deltamonth = abs(float(transaction1[8])-float(currenttransaction[8]))
        if deltamonth > 0:
            deltadays = 0
        else:
            deltadays = abs(float(transaction1[9])-float(currenttransaction[9]))
        for transaction2 in alltransactions:
            if (transaction2[2] == transaction1[2]) or (transaction2[2] == currenttransaction[2]):
                continue
            if (abs(float(transaction2[8])-float(transaction1[8])) == deltamonth) and (abs(float(transaction2[9])-float(transaction1[9])) == deltadays):
                return True
            elif (abs(float(transaction2[8])-float(currenttransaction[8])) == deltamonth) and (abs(float(transaction2[9])-float(currenttransaction[9])) == deltadays):
                return True
    return False

def createallTransaction():
    with open('transactions_clean.txt', 'r') as myfile:
        alltransactions=[]
        lines = myfile.read()
        while isNotEmpty(lines):
            currenttransactions=[]
            try:
                _id= find_between(lines,"_id",",")
                _id=_id.split("\"")[2]
                lines= lines[lines.find(","):]
            except ValueError:
                break

            category=find_between(lines,"[","]")
            #  need to use category to list
            lines = lines[lines.find("]"):]

            id = find_between(lines, "id", ",")
            id = id.split("\"")[2]
            lines = lines[lines.find(","):]

            userId = find_between(lines, "userId", ",")
            userId = userId.split("\"")[2]
            lines = lines[lines.find(","):]

            accountId = find_between(lines, "accountId", ",")
            accountId = accountId.split("\"")[2]
            lines = lines[lines.find(","):]

            amount = find_between(lines, "amount", ",")
            amount = float(amount.split(":")[1])
            lines = lines[lines.find(","):]

            categoryId = find_between(lines, "categoryId", ",")
            categoryId = categoryId.split("\"")[2]
            lines = lines[lines.find(","):]

            date = find_between(lines, "date", ",")
            date = date.split("\"")[2]
            date=map(lambda x:int(x),date.split("-"))
            year=float(date[0])
            month=float(date[1])
            day=float(date[2])
            lines = lines[lines.find(","):]

            location = find_between(lines, "{", "}")
            # need to use composite to list
            lines = lines[lines.find("}"):]

            name = find_between(lines, "name", ",")
            name = name.split("\"")[2]
            lines = lines[lines.find(","):]

            paymentMeta = find_between(lines, "{", "}")
            #  need to use composite to list
            lines = lines[lines.find("}"):]

            creditCardTransaction = find_between(lines, "creditCardTransaction", ",")
            creditCardTransaction = creditCardTransaction.split(":")[1].strip() == "true"
            lines = lines[lines.find(","):]

            subscription = find_between(lines, "subscription", ",")
            subscription = subscription.split("\"")[2]
            lines = lines[lines.find("}"):]

            currenttransactions.append(_id)
            currenttransactions.append(category)
            currenttransactions.append(id)
            currenttransactions.append(userId)
            currenttransactions.append(accountId)
            currenttransactions.append(amount)
            currenttransactions.append(categoryId)
            currenttransactions.append(year)
            currenttransactions.append(month)
            currenttransactions.append(day)
            currenttransactions.append(location)
            currenttransactions.append(name)
            currenttransactions.append(paymentMeta)
            currenttransactions.append(creditCardTransaction)
            currenttransactions.append(subscription)

            alltransactions.append(currenttransactions)
        alltransactions=np.asarray(alltransactions)

        predictedWeekly=np.apply_along_axis(lambda x:predictweek(x,alltransactions), 1, alltransactions)
        alltransactions=np.c_[alltransactions, predictedWeekly]

        predictedMonthly = np.apply_along_axis(lambda x: predictmonth(x, alltransactions), 1, alltransactions)
        alltransactions = np.c_[alltransactions, predictedMonthly]

        predictedsubscription = np.apply_along_axis(lambda x: predictsubscription(x, alltransactions), 1, alltransactions)
        alltransactions = np.c_[alltransactions, predictedsubscription]
        return alltransactions

def compositeToList(compositeStr):
    return [x.split(":")[1].strip() for x in compositeStr.split(",")]

def categoryToList(category):
    return [x.split("\"")[1] for x in category.split(",")]

createallTransaction()

def getmodel(alltransaction):
    return 0
    # input = alltransaction[:, 0:14]
    # outputweak = alltransaction[:, 15]
    # outputmonth = alltransaction[:, 16]
    # outputsubscription = alltransaction[:, 17]
    #
    #
    # model = Sequential()
	# model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(1, kernel_initializer='normal'))
	# model.compile(loss='mean_squared_error', optimizer='adam')
	# return model


