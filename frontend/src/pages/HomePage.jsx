import { Link } from 'react-router-dom'
import './HomePage.css'

export default function HomePage() {
  return (
    <div className="home">
      <section className="hero">
        <div className="hero-content">
          <p className="hero-tagline">A Global Lens on Local Taste</p>
          <h1 className="hero-title">
            How McDonald's Menus
            <br />
            <span className="highlight">Mirror Culture</span>
          </h1>
          <p className="hero-description">
            Explore how menu variations across 100+ countries reflect cultural preferences,
            regional flavors, and the fascinating ways a global brand adapts to local palates.
          </p>
          <Link to="/explore" className="cta-button">
            Explore the Data
          </Link>
        </div>
        <div className="hero-visual">
          <div className="globe-placeholder">
            <span className="globe-icon">🌍</span>
            <span className="globe-label">100+ Countries</span>
          </div>
        </div>
      </section>

      <section className="features">
        <h2 className="section-title">What We Analyze</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <div className="feature-icon">🍔</div>
            <h3>Menu Variations</h3>
            <p>Compare Big Macs, McSpicy, Teriyaki, and hundreds of regional exclusives.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🌏</div>
            <h3>Cultural Patterns</h3>
            <p>Discover how spice levels, ingredients, and formats correlate with cultural norms.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Data Insights</h3>
            <p>Evidence-based analysis linking menu choices to Hofstede dimensions and more.</p>
          </div>
        </div>
      </section>

      <section className="cta-section">
        <h2>Ready to explore?</h2>
        <p>Dive into our interactive analysis of McDonald's menus worldwide.</p>
        <Link to="/explore" className="cta-button secondary">
          Start Exploring
        </Link>
      </section>
    </div>
  )
}
