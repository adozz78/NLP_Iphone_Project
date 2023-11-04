class Model:
    def __init__(self, X, y, model_architecture, vectorizer, random_seed=42, test_size=0.2) -> None:
        self.X = X
        self.y = y
        self.model_instance = model_architecture
        self.vectorizer = vectorizer
        self.random_seed = random_seed
        self.test_size = test_size

        self.pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model_architecture)
        ]) 

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed)
    

    
    def fit(self):
        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self):
        return self.pipeline.predict(self.X_test)

    
    def predict_proba(self):
        return self.pipeline.predict_proba(self.X_test)
   
    
    def report(self, y_true, y_pred, class_labels):
        print(classification_report(y_true, y_pred, labels=class_labels, zero_division=0))

        confusion_matrix_kwargs = dict(
            text_auto=True,
            title="Confusion Matrix",
            width=1000,
            height=800,
            labels=dict(x="Predicted", y="True Label"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Blues'
        )
        
        c_m = confusion_matrix(y_true, y_pred, labels=class_labels) 
        fig = px.imshow(c_m, **confusion_matrix_kwargs)
        fig.show()