import React, { useEffect, useState } from 'react'
import axios from 'axios'

interface Props {
  item: string
  category: string
  onItemChange: (item: string) => void
  onCategoryChange: (category: string) => void
}

const ItemCategorySelector: React.FC<Props> = ({ item, category, onItemChange, onCategoryChange }) => {
  const [categories, setCategories] = useState<string[]>([])
  const [items, setItems] = useState<string[]>([])

  // fetch categories on mount
  useEffect(() => {
    axios.get('/api/categories')
      .then(res => setCategories(res.data))
      .catch(err => console.error(err))
  }, [])

  // fetch items when category changes
  useEffect(() => {
    if (category) {
      axios.get(`/api/items?category=${encodeURIComponent(category)}`)
        .then(res => setItems(res.data))
        .catch(err => console.error(err))
    } else {
      // all items
      axios.get('/api/items')
        .then(res => setItems(res.data))
        .catch(err => console.error(err))
    }
  }, [category])

  return (
    <div className="control-panel">
      <label htmlFor="category-select">분류: </label>
      <select id="category-select" value={category} onChange={e => onCategoryChange(e.target.value)}>
        <option value="">전체</option>
        {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
      </select>
      <label htmlFor="item-select">품목: </label>
      <select id="item-select" value={item} onChange={e => onItemChange(e.target.value)}>
        <option value="">전체</option>
        {items.map(it => <option key={it} value={it}>{it}</option>)}
      </select>
    </div>
  )
}

export default ItemCategorySelector