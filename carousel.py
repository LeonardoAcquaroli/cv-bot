from streamlit_carousel import carousel

carousel_items = [
    dict(
        title="iDays win 2022",
        text="The Quokkay team wins the iDays 2022 Hackathon",
        img="https://raw.githubusercontent.com/LeonardoAcquaroli/cv-bot/main/images/quokkay.jpeg",
        link="https://discuss.streamlit.io/t/new-component-react-bootstrap-carousel/46819"
    ),
    dict(
        title="Big data for Gender equality",
        text="Me and Melissa, presenting our project in Auditorium Cascina Triulza at MIND in Milan",
        img="https://raw.githubusercontent.com/LeonardoAcquaroli/cv-bot/main/images/mind.jpeg",
        link="https://github.com/thomasbs17/streamlit-contributions/tree/master"
    ),
    dict(
        title="Playing football",
        text="This is me during a summer tournament in 2022",
        img="https://raw.githubusercontent.com/LeonardoAcquaroli/cv-bot/main/images/football.jpg"
    ),
]

leo_carousel = carousel(items=carousel_items, width=1)