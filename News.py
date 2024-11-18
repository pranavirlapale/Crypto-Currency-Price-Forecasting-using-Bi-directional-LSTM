import streamlit as st
import requests

def app():
    st.title("Cryptocurrency News")

    # Define the API URL
    api_url = "https://newsdata.io/api/1/news?apikey=pub_31232a17ff003844da601608d829737bd8380&q=Cryptocurrency&language=en"

    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("results", [])

            for index, article in enumerate(articles):
                title = article.get("title", "")
                description = article.get("description", "")
                source = article.get("creator", "")
                published_at = article.get("pubDate", "")
                url = article.get("Continue Reading", "")
                image_url = article.get("image_url", "")  # Get the image URL

                col1, col2 = st.columns([1, 2])  # Split the layout into two columns

                # Display the image on the left with adjusted size if image_url is not null
                with col1:
                    if image_url:
                        st.image(image_url, caption="Image", use_column_width='auto', width=50)

                # Display title and description on the right
                with col2:
                    st.subheader(str(title))
                    st.write(str(description))
                    st.write("Source: " + str(source) + " - Published: " + str(published_at))

                    # Use a unique key for each button
                    button_key = f"read_more_{index}"
                    if st.button("Read More", key=button_key):
                        st.markdown("Read the full article: " + str(url))
                st.markdown('---')
        else:
            st.warning(f"Failed to fetch cryptocurrency news. Status code: {response.status_code}")
    except Exception as e:
        st.warning(f"An error occurred while fetching cryptocurrency news: {str(e)}")

if __name__ == "__main__":
    app()
