# Skyscanner Internal Blog Post Structure
**Title Idea:** *Beyond the Flight: Using Gradient Boosting to Predict Package Upsells Without Annoying the User*

## 1. The "Why" (The Business Context)
*   Why are we talking about packages? (Higher margins, locking the customer into our ecosystem early).
*   The UX Dilemma: The "spray and pray" approach of showing a package upsell popup to *every* flight searcher causes banner blindness and UI friction. We need to be smart about *who* sees the offer.

## 2. The Data (What signals intent?)
*   Briefly explain what features we track. Do they have kids? How far in advance are they booking? How far are they traveling?
*   *Skyscanner context:* Mention how we could also use flight duration or destination weather as features.

## 3. The Algorithm (PM's Guide to Gradient Boosting)
*   Explain the math simply: **Gradient Descent** is like playing golf. Your first shot (base model) gets you close. Your second shot (first tree) aims only to correct the remaining distance to the hole (the residual error). You keep taking smaller and smaller putts (learning rate) until you sink it. 
*   Explain why we don't just use simple Linear regression (because humans don't make linear decisions).

## 4. The Trade-offs: Precision vs. Recall (The most important PM section)
*   **If we optimize for Recall:** We cast a wide net. We catch almost everyone who might want a package, but we annoy a lot of people who don't (False Positives).
*   **If we optimize for Precision:** We only show the offer when we are highly confident. We annoy very few people, but we leave money on the table (False Negatives).
*   *Your PM thesis:* For a flight search UX (which is already stressful), we should lean heavily toward **Precision** to protect the user experience. 

## 5. Call to Action
*   How can other PMs or engineers at Skyscanner start thinking about contextual UI elements in their pods?
