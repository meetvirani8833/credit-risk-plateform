import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Chat from './pages/Chat'
import DataOverview from './pages/DataOverview'
import Explainability from './pages/Explainability'
import Prediction from './pages/Prediction'
import Rules from './pages/Rules'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DataOverview />} />
          <Route path="prediction" element={<Prediction />} />
          <Route path="explainability" element={<Explainability />} />
          <Route path="rules" element={<Rules />} />
          <Route path="chat" element={<Chat />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
