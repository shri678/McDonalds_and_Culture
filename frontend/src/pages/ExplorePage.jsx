import { useState } from 'react'
import './ExplorePage.css'

// Sample data for demonstration - will connect to backend API later
const SAMPLE_COUNTRIES = [
  { code: 'US', name: 'United States', flag: '🇺🇸' },
  { code: 'JP', name: 'Japan', flag: '🇯🇵' },
  { code: 'IN', name: 'India', flag: '🇮🇳' },
  { code: 'DE', name: 'Germany', flag: '🇩🇪' },
  { code: 'BR', name: 'Brazil', flag: '🇧🇷' },
  { code: 'KR', name: 'South Korea', flag: '🇰🇷' },
]

const SAMPLE_ITEMS = {
  US: ['Big Mac', 'McChicken', 'Quarter Pounder', 'McRib', 'Filet-O-Fish'],
  JP: ['Teriyaki McBurger', 'McPork', 'Ebi Filet-O', 'Tsukimi Burger', 'McChicken'],
  IN: ['McAloo Tikki', 'McSpicy Paneer', 'McVeggie', 'Maharaja Mac', 'McChicken'],
  DE: ['Big Mac', 'McRib', 'Chickenburger', 'McWrap', 'Currywurst'],
  BR: ['Big Mac', 'Cheddar McMelt', 'McNífico Bacon', 'Quarteirão', 'McFritas'],
  KR: ['Bulgogi Burger', 'Shrimp Burger', '1955 Burger', 'McSpicy', 'McRice'],
}

export default function ExplorePage() {
  const [selectedCountry, setSelectedCountry] = useState('US')

  return (
    <div className="explore">
      <div className="explore-header">
        <h1>Explore Menus by Country</h1>
        <p>Select a country to view regional menu items and cultural insights.</p>
      </div>

      <div className="explore-content">
        <aside className="country-sidebar">
          <h3>Countries</h3>
          <ul className="country-list">
            {SAMPLE_COUNTRIES.map((country) => (
              <li key={country.code}>
                <button
                  className={`country-btn ${selectedCountry === country.code ? 'active' : ''}`}
                  onClick={() => setSelectedCountry(country.code)}
                >
                  <span className="country-flag">{country.flag}</span>
                  <span className="country-name">{country.name}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <div className="main-panel">
          <div className="country-detail">
            <div className="country-header">
              <span className="detail-flag">{SAMPLE_COUNTRIES.find(c => c.code === selectedCountry)?.flag}</span>
              <h2>{SAMPLE_COUNTRIES.find(c => c.code === selectedCountry)?.name}</h2>
            </div>
            <div className="menu-section">
              <h3>Signature Menu Items</h3>
              <ul className="menu-list">
                {(SAMPLE_ITEMS[selectedCountry] || SAMPLE_ITEMS.US).map((item, i) => (
                  <li key={i} className="menu-item">
                    <span className="menu-icon">🍔</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="insight-box">
              <h4>Cultural Insight</h4>
              <p>
                {selectedCountry === 'US' && "The US menu emphasizes classic beef burgers and familiar staples, reflecting a preference for consistency and tradition."}
                {selectedCountry === 'JP' && "Japan's menu features seasonal items like the Tsukimi Burger (moon-viewing) and local flavors like teriyaki, blending global branding with Japanese culinary culture."}
                {selectedCountry === 'IN' && "India offers a largely vegetarian-focused menu with paneer and potato-based options, reflecting dietary preferences and religious considerations."}
                {selectedCountry === 'DE' && "Germany's menu includes the McRib year-round and Currywurst, aligning with local meat-forward preferences and popular street food."}
                {selectedCountry === 'BR' && "Brazil showcases cheese-heavy options like Cheddar McMelt, mirroring the local love for queijo and bold flavors."}
                {selectedCountry === 'KR' && "South Korea features Bulgogi-inspired burgers and rice-based options, integrating traditional Korean flavors into the global format."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
