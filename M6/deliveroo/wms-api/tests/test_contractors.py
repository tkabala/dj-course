import json


class TestGetContractors:
    def test_returns_seeded_contractors(self, client):
        response = client.get('/contractors/')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_deleted_contractor_not_in_list(self, client):
        client.delete('/contractors/2')
        data = client.get('/contractors/').get_json()
        assert not any(c['id'] == '2' for c in data)


class TestDeleteContractor:
    def test_delete_returns_200(self, client):
        response = client.delete('/contractors/1')
        assert response.status_code == 200
        assert 'deleted successfully' in response.get_json()['message']

    def test_deleted_contractor_not_returned_in_list(self, client):
        client.delete('/contractors/1')
        data = client.get('/contractors/').get_json()
        assert not any(c['id'] == '1' for c in data)

    def test_delete_nonexistent_returns_404(self, client):
        assert client.delete('/contractors/9999').status_code == 404

    def test_double_delete_returns_404(self, client):
        client.delete('/contractors/1')
        assert client.delete('/contractors/1').status_code == 404


class TestPatchContractorStatus:
    def test_patch_status_returns_200(self, client):
        response = client.patch(
            '/contractors/1',
            data=json.dumps({'status': 'INACTIVE'}),
            content_type='application/json'
        )
        assert response.status_code == 200

    def test_updated_status_visible_in_list(self, client):
        client.patch('/contractors/1',
                     data=json.dumps({'status': 'INACTIVE'}),
                     content_type='application/json')
        data = client.get('/contractors/').get_json()
        contractor = next(c for c in data if c['id'] == '1')
        assert contractor['status'] == 'INACTIVE'

    def test_patch_nonexistent_returns_404(self, client):
        response = client.patch('/contractors/9999',
                                data=json.dumps({'status': 'ACTIVE'}),
                                content_type='application/json')
        assert response.status_code == 404

    def test_patch_invalid_status_returns_400(self, client):
        response = client.patch('/contractors/1',
                                data=json.dumps({'status': 'BANANA'}),
                                content_type='application/json')
        assert response.status_code == 400
