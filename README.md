# rr-AI-Trainer
The files used for the pilot study in my COMP2121 Cwk.

## Usage Instructions

Follow these steps to generate book recommendations:

1. **Search for Books**  
   Run `search.py` to retrieve book URLs.  
   You can adjust the number of pages (20 books per page) by modifying the `NUMBER_OF_PAGES` variable at the top of the script:
   ```bash
   python search.py
   ```

2. **Scrape Book Blurbs**  
   Once you have the URLs, run `scrape.py` to extract the blurbs:
   ```bash
   python scrape.py
   ```

3. **Process the Data**  
   Process the scraped data by running `process_data.py`.  
   You can set how many books to include by modifying the `NUMBER_OF_BOOKS` variable:
   ```bash
   python process_data.py
   ```

4. **Fine-Tune the Model**  
   Run `fine_tune_model.py` to train the recommendation model.  
   Again, you can adjust `NUMBER_OF_BOOKS` if needed:
   ```bash
   python fine_tune_model.py
   ```

5. **Get Recommendations**  
   Use `recommend.py` to generate book recommendations, you will have to manually call the function.

   Optionally, the repository is configured to support running `host_model.py` with FastAPI for serving the model via an API.
