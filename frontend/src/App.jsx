import React from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';

function App() {
  const scrollToWorkspace = () => {
    document.querySelector('#workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="app-shell">
      <Navbar onRunAnalysis={scrollToWorkspace} />
      <main><Dashboard /></main>
      <footer>
        <div className="footer-brand">pricely<span>.</span></div>
        <p>Make every price a confident decision.</p>
        <span>© {new Date().getFullYear()} Pricely Analytics</span>
      </footer>
    </div>
  );
}

export default App;
