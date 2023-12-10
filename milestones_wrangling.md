Milestones for wrangling stage.


 - In the data, there is no clear borderline for some of the ordinal and continuous features. We set an arbitrary rule to distinguish based on the number of levels(n = 50 levels cutoff).  This allows for an organized strategy in treating and understanding the features. 
 
 - Outliers were checked for continuous features. They were not handled in the original dataset due to our lack of domain knowledge. But during the modeling phase we discussed and created an additional dataset  with outliers removed,  to fit the model and compare the accuracy with initial models. 
 
 -Initially, variables in test set were scaled using test mean and standard deviation. During the modeling phase,  we reflected on the practice and decided to create a set of new scaled variables using the statistics in training set. 