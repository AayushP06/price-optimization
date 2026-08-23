import React from 'react';

const Footer = () => {
  return (
    <footer style={{ borderTop: '1px solid #334155', paddingTop: '3rem', paddingBottom: '2rem', marginTop: '4rem', background: '#0f172a' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 2rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '2rem' }}>
        
        <div style={{ flex: 1, minWidth: '250px' }}>
          <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#f8fafc', marginBottom: '1rem' }}>
            <span style={{ color: '#6366f1' }}>Price</span>Optimizer
          </div>
          <p style={{ color: '#94a3b8', lineHeight: '1.6' }}>
            B2B SaaS pricing intelligence. Helping modern commerce businesses maximize their profit margins through data-driven decisions.
          </p>
        </div>

        <div>
          <h4 style={{ color: '#f8fafc', marginBottom: '1rem' }}>Developer</h4>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <li>
              <a href="https://github.com/AayushP06" target="_blank" rel="noopener noreferrer" style={{ color: '#94a3b8', textDecoration: 'none' }}>
                GitHub Profile
              </a>
            </li>
            <li>
              <a href="https://www.linkedin.com/in/aayush-prasad-78672a329" target="_blank" rel="noopener noreferrer" style={{ color: '#94a3b8', textDecoration: 'none' }}>
                LinkedIn Network
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 2rem', marginTop: '3rem', paddingTop: '1.5rem', borderTop: '1px solid #1e293b', textAlign: 'center', color: '#64748b', fontSize: '0.875rem' }}>
        <p>© {new Date().getFullYear()} Built by Aayush Prasad. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;