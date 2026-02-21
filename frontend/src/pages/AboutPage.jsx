import './AboutPage.css'

export default function AboutPage() {
  return (
    <div className="about">
      <div className="about-content">
        <h1>About This Project</h1>
        <p className="about-lead">
          McDonald's & Culture explores the fascinating relationship between global fast food
          menus and cultural diversity across the world.
        </p>

        <section className="about-section">
          <h2>The Question</h2>
          <p>
            How does a single global brand adapt its menu to appeal to over 100 countries?
            What can we learn about culture by analyzing the differences—and similarities—in
            what McDonald's offers from Tokyo to São Paulo?
          </p>
        </section>

        <section className="about-section">
          <h2>Our Approach</h2>
          <p>
            We collect and analyze menu data from McDonald's restaurants worldwide, comparing
            ingredients, flavor profiles, and regional exclusives. By cross-referencing this
            data with cultural frameworks—such as Hofstede's dimensions, culinary traditions,
            and dietary norms—we uncover patterns that reveal how taste and identity intersect.
          </p>
        </section>

        <section className="about-section">
          <h2>What You'll Find</h2>
          <ul>
            <li><strong>Menu comparisons</strong> across countries and regions</li>
            <li><strong>Cultural insights</strong> tied to specific menu variations</li>
            <li><strong>Data visualizations</strong> of trends and correlations</li>
            <li><strong>Research methodology</strong> and sources for further exploration</li>
          </ul>
        </section>

        <section className="about-section">
          <h2>Get in Touch</h2>
          <p>
            This project is an ongoing analysis. If you have data, corrections, or ideas to
            contribute, we'd love to hear from you.
          </p>
        </section>
      </div>
    </div>
  )
}
